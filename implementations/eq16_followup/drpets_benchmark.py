"""Reproducible, independent PETS vs parameter-W2 DR-PETS research prototype.

This is NOT a port or reproduction of the authors' GitHub code or results.
It implements a parametric W2 interpretation of the p=2 gradient regularizer
from Jesawada et al., DR-PETS (2025), arXiv:2503.20660, Eq. (7)/(16).

Dependencies: torch, numpy, pandas, matplotlib. No Gymnasium required.
Original CartPole and Pendulum equations follow Gym's reference dynamics,
implemented independently; no official Gymnasium validation is claimed.

Usage:
 python drpets_benchmark.py --task cartpole --quick --out results/cartpole
 python drpets_benchmark.py --task pendulum --quick --out results/pendulum
 python drpets_benchmark.py --task both --out results/full
 python drpets_benchmark.py --self-test
"""
from __future__ import annotations
import argparse, copy, json, math, os, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CPU inference dominates benchmarking; small matrices run faster with one thread.
torch.set_num_threads(1)
DTYPE = torch.float32


class CartPole:
    """CartPole-v1-like continuous physical state, with discrete {0,1} actions."""
    name='cartpole'; dim=4; act_dim=1; discrete=True; horizon=200
    def __init__(self, pole_length=0.5, seed=0):
        self.length=float(pole_length) # HALF pole length, as in classic CartPole
        self.rng=np.random.default_rng(seed)
    def reset(self):
        self.state=self.rng.uniform(-0.05,0.05,size=4).astype('float32')
        return self.state.copy()
    def step(self,action):
        x,x_dot,theta,theta_dot=map(float,self.state)
        force=10.0 if int(action)==1 else -10.0
        costheta,sintheta=math.cos(theta),math.sin(theta)
        total_mass=1.1
        pm_len=0.1*self.length
        temp=(force + pm_len*theta_dot**2*sintheta)/total_mass
        thetaacc=(9.8*sintheta-costheta*temp)/(self.length*(4.0/3.0-0.1*costheta**2/total_mass))
        xacc=temp-pm_len*thetaacc*costheta/total_mass
        dt=0.02
        self.state=np.array([x+dt*x_dot,x_dot+dt*xacc,theta+dt*theta_dot,theta_dot+dt*thetaacc],dtype='float32')
        failed=abs(self.state[0])>2.4 or abs(self.state[2])>12*math.pi/180
        return self.state.copy(),1.0,bool(failed)
    def action(self,a): return int(a)
    def reward(self,state,actions):
        # The planner uses a smooth, known surrogate cost, identical for PETS and DR-PETS.
        x,dx,th,dth=state.unbind(-1)
        cost=1.8*x.square()+0.18*dx.square()+20*th.square()+0.2*dth.square()
        invalid=(x.abs()>2.4)|(th.abs()>12*math.pi/180)
        return -cost-12.0*invalid.to(state.dtype)
    def features(self,state,actions):
        return torch.cat((state / torch.tensor([2.4,3.,0.25,3.],dtype=DTYPE), (2*actions-1)), dim=-1)


class Pendulum:
    """Pendulum-v1-like torque-controlled swing-up. State: [angle (unwrapped), angular velocity]."""
    name='pendulum'; dim=2; act_dim=1; discrete=False; horizon=200
    def __init__(self, mass=1.0, length=1.0, seed=0):
        self.mass=float(mass);self.length=float(length);self.rng=np.random.default_rng(seed)
    def reset(self):
        self.state=np.array([self.rng.uniform(-math.pi,math.pi),self.rng.uniform(-1,1)],dtype='float32')
        return self.state.copy()
    def step(self,action):
        u=float(np.clip(action,-2,2)); th,vel=map(float,self.state)
        newvel=vel+(3*10/(2*self.length)*math.sin(th)+3/(self.mass*self.length**2)*u)*0.05
        newvel=np.clip(newvel,-8,8); newth=th+newvel*0.05
        self.state=np.array([newth,newvel],dtype='float32')
        cost=((th+math.pi)%(2*math.pi)-math.pi)**2+0.1*vel**2+0.001*u**2
        return self.state.copy(),-float(cost),False
    def action(self,a): return float(np.clip(a,-2,2))
    def reward(self,state,actions):
        th,omega=state.unbind(-1)
        angle=torch.atan2(torch.sin(th),torch.cos(th))
        return -(angle.square()+0.1*omega.square()+0.001*actions.squeeze(-1).square())
    def features(self,state,actions):
        th,vel=state.unbind(-1)
        return torch.stack((torch.cos(th),torch.sin(th),vel/8,actions.squeeze(-1)/2),dim=-1)


def make_env(task, shift, seed):
    if task=='cartpole': return CartPole(pole_length=shift,seed=seed)
    if task=='pendulum': return Pendulum(mass=shift,seed=seed)
    raise ValueError(task)


class GaussianDeltaModel(nn.Module):
    def __init__(self, feature_dim, state_dim, width=48):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(feature_dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,2*state_dim))
        self.state_dim=state_dim
        self.register_buffer('delta_mean',torch.zeros(state_dim))
        self.register_buffer('delta_std',torch.ones(state_dim))
    def forward(self,features):
        v=self.net(features)
        mean=v[...,:self.state_dim]
        logstd=v[...,self.state_dim:].clamp(-4.5,0.6)
        return mean,logstd
    def sample_next(self,state,features):
        mean,logstd=self(features)
        noise=torch.randn_like(mean)
        delta=(mean+logstd.exp()*noise)*self.delta_std+self.delta_mean
        next_state=(state+delta).detach()
        next_target=((next_state-state.detach()-self.delta_mean)/self.delta_std).detach()
        logprob=torch.distributions.Normal(mean,logstd.exp()).log_prob(next_target).sum(-1)
        return next_state,logprob


def collect_data(env, n, seed):
    rng=np.random.default_rng(seed);xs=[];us=[];ds=[]
    env.reset()
    for k in range(n):
        # Reset regularly to cover both near-equilibrium and wider trajectories.
        if k%55==0: env.reset()
        a=rng.integers(0,2) if env.discrete else rng.uniform(-2,2)
        old=env.state.copy();nxt,_,failed=env.step(a)
        xs.append(old);us.append([float(a)]);ds.append(nxt-old)
        if failed: env.reset()
    return (torch.tensor(np.array(xs),dtype=DTYPE),torch.tensor(np.array(us),dtype=DTYPE),torch.tensor(np.array(ds),dtype=DTYPE))


def train_ensemble(env, data, seed=0, B=3, epochs=20, width=48, batch_size=256):
    x,u,delta=data; inputs=env.features(x,u)
    mean=delta.mean(0);std=delta.std(0).clamp_min(1e-3)
    norm_delta=(delta-mean)/std
    models=[];n=len(x)
    for b in range(B):
        torch.manual_seed(seed+1009*b)
        model=GaussianDeltaModel(inputs.shape[-1],env.dim,width)
        model.delta_mean.copy_(mean);model.delta_std.copy_(std)
        opt=torch.optim.Adam(model.parameters(),lr=0.002)
        rng=np.random.default_rng(seed+11*b)
        idx=torch.tensor(rng.integers(0,n,size=n),dtype=torch.long) # bootstrap
        for epoch in range(epochs):
            order=idx[torch.randperm(n)]
            for sel in order.split(batch_size):
                mu,logstd=model(inputs[sel])
                neglog=0.5*((norm_delta[sel]-mu).square()*torch.exp(-2*logstd)+2*logstd).sum(-1).mean()
                opt.zero_grad();neglog.backward();opt.step()
        model.eval();models.append(model)
    return models


def score_candidates(env,models,sequences,epsilon=0.,Q=6,gamma=0.98,seed=0,
                     initial_state=None,return_parts=False):
    """Expected reward & exact neural-parameter score-function sensitivity.

    Each candidate has Q independent particles per ensemble member; each model stays
    fixed throughout each particle rollout (PETS TS-infinity convention).
    Gradient of expected return uses the likelihood-ratio identity:
       grad_theta J = E[sum_k grad_theta log f_theta(x_{k+1}|x_k,u_k) * G_{k+1}]
    A leave-one-particle-out baseline reduces variance without adding bias.
    We use batched VJPs to obtain an M x |theta| gradient for M candidates.
    The parameter vector is NOT updated during planning.
    """
    N,H=sequences.shape[:2]
    initial=torch.tensor(initial_state,dtype=DTYPE).reshape(1,1,-1)
    reward_ensemble=[];norm2_ensemble=[]
    with torch.enable_grad():
        for b,model in enumerate(models):
            torch.manual_seed(seed+997*b)
            s=initial.expand(N,Q,-1).clone().detach()
            rewards=[];logprobs=[]
            for t in range(H):
                act=sequences[:,t,:].unsqueeze(1).expand(N,Q,-1)
                rewards.append(env.reward(s,act).detach())
                if t<H-1:
                    s,logp=model.sample_next(s,env.features(s,act))
                    logprobs.append(logp)
            r=torch.stack(rewards,dim=-1) # N,Q,H
            disc=torch.tensor([gamma**i for i in range(H)],dtype=DTYPE)
            discounted=r*disc
            total=discounted.sum(-1)   # N,Q
            reward_ensemble.append(total.mean(1))
            if epsilon>0:
                # At time k, only rewards at k+1,...,H-1 depend on transition k.
                downstream=torch.flip(torch.cumsum(torch.flip(discounted,[-1]),-1),[-1])[...,1:]
                lp=torch.stack(logprobs,dim=-1)
                # Leave-one-out baseline; unbiased for independent particles, Q>=2.
                baseline=(downstream.sum(dim=1,keepdim=True)-downstream)/(Q-1)
                objective=((downstream-baseline).detach()*lp).sum(dim=(1,2))/Q  # [N]
                params=tuple(p for p in model.parameters() if p.requires_grad)
                g=torch.autograd.grad(objective,params,
                      grad_outputs=torch.eye(N,dtype=DTYPE),is_grads_batched=True)
                norm2=sum(v.flatten(start_dim=1).square().sum(-1) for v in g)
                norm2_ensemble.append(norm2.detach())
    nominal=torch.stack(reward_ensemble).mean(0).detach()
    penalty=torch.sqrt(torch.stack(norm2_ensemble).mean(0).clamp_min(0)) if epsilon>0 else torch.zeros_like(nominal)
    value=nominal-epsilon*penalty
    if return_parts: return value,nominal,penalty
    return value


def plan(env,models,state,method,epsilon,seed,quick=False):
    H=8 if quick else 15
    M=12 if quick else 48
    Q=4 if quick else 12
    iterations=2 if quick else 4
    K=max(3,M//5)
    rng=np.random.default_rng(seed)
    best_value=-np.inf; best_action=None
    if env.discrete:
        probs=np.ones(H,dtype=float)*0.5
    else:
        means=np.zeros(H,dtype=float);stds=np.ones(H,dtype=float)*1.3
    for it in range(iterations):
        if env.discrete:
            a=(rng.random((M,H))<probs).astype('float32')
        else:
            a=np.clip(rng.normal(means,stds,(M,H)),-2,2).astype('float32')
        # Always evaluate a neutral candidate, independent of the random CEM distribution.
        a[0,:]=1.0 if env.discrete else 0.0
        seq=torch.tensor(a[:,:,None],dtype=DTYPE)
        scores=score_candidates(env,models,seq,epsilon if method=='drpets' else 0.,Q,seed=seed+it*10000,initial_state=state)
        score=scores.numpy();elite=np.argsort(score)[-K:]
        idx=int(score.argmax())
        if score[idx]>best_value:
            best_value=float(score[idx]);best_action=a[idx,0]
        if env.discrete:
            probs=np.clip(0.25*probs+0.75*a[elite].mean(0),0.06,0.94)
        else:
            means=0.25*means+0.75*a[elite].mean(0)
            stds=np.clip(0.25*stds+0.75*a[elite].std(0),0.2,2.0)
    return env.action(best_action)


def evaluate(env_name, models, shift, method, eps, seeds, quick=False, steps=None):
    result=[]
    for seed in seeds:
        env=make_env(env_name,shift,seed+7000)
        state=env.reset();total=0.;survived=0
        max_steps=steps or (100 if quick else 200)
        for t in range(max_steps):
            act=plan(env,models,state,method,eps,seed=seed*100000+t*431,quick=quick)
            state,reward,done=env.step(act)
            total+=reward;survived+=1
            if done: break
        result.append(dict(task=env_name,shift=shift,method=method,epsilon=eps,seed=seed,
                           return_value=total,steps=survived))
    return result


def benchmark(task, quick, out, epsilon, seed, eval_steps=None, eval_seeds=None, shifts=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    nominal=0.5 if task=='cartpole' else 1.
    shifted=0.8 if task=='cartpole' else 1.5
    env=make_env(task,nominal,seed)
    n=850 if quick else 4500
    epochs=10 if quick else 40
    data=collect_data(env,n,seed)
    start=time.time();models=train_ensemble(env,data,seed,B=3,epochs=epochs,width=48)
    print(f'{task}: trained B=3 on {n} transitions in {time.time()-start:.1f}s',flush=True)
    # Validate basic gradient behavior on same model, fixed candidate set
    x0=env.reset()
    seq=torch.zeros((2,5,1),dtype=DTYPE)
    if env.discrete: seq[0,:,:]=1
    v,nom,p=score_candidates(env,models,seq,epsilon=epsilon,Q=6,initial_state=x0,return_parts=True)
    print('  score-sanity nominal',nom.tolist(),'gradient norms',p.tolist(),flush=True)
    results=[]
    seeds=[seed+19+10*j for j in range(eval_seeds if eval_seeds is not None else (2 if quick else 5))]
    for shift in (shifts if shifts is not None else [nominal,shifted]):
        for method in ['pets','drpets']:
            t0=time.time()
            records=evaluate(task,models,shift,method,epsilon,seeds,quick=quick,steps=eval_steps)
            results+=records
            print('  ',task,'shift',shift,method,'returns',[round(x['return_value'],2) for x in records],f'time {time.time()-t0:.1f}s',flush=True)
            pd.DataFrame(results).to_csv(out/'raw_results.csv',index=False)
    df=pd.DataFrame(results)
    df.to_csv(out/'raw_results.csv',index=False)
    summary=df.groupby(['task','shift','method']).agg(n=('return_value','count'),mean_return=('return_value','mean'),std_return=('return_value','std'),mean_steps=('steps','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    for method,subset in summary.groupby('method'):
        x=subset['shift'].to_numpy();y=subset['mean_return'].to_numpy();sd=subset['std_return'].fillna(0).to_numpy()
        ax.errorbar(x,y,yerr=sd,marker='o',capsize=4,label=method.upper())
    ax.set_xlabel('Half pole length (m)' if task=='cartpole' else 'Pendulum mass (kg)')
    ax.set_ylabel('Measured episodic reward')
    ax.set_title(f'{task.title()}: PETS vs parameter-gradient DR-PETS'+ (' (smoke test)' if quick else ''))
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    torch.save([m.state_dict() for m in models],out/'trained_ensemble.pt')
    with open(out/'metadata.json','w') as f:
        json.dump(dict(task=task,quick=quick,data=n,train_epochs=epochs,num_models=3,
                       epsilon=epsilon,seeds=seeds,parameter_W2_metric='Euclidean NN weights',
                       gradient_estimator='per-candidate REINFORCE score, downstream returns, leave-one-out baseline',
                       planner='CEM, TS infinity, separate ensemble member particle rollouts',
                       gymnasium_installed=False,official_benchmark=False,nominal=nominal,shifted=shifted,
                       eval_steps=eval_steps,shifts=(shifts if shifts is not None else [nominal,shifted])),f,indent=2)
    return summary


def self_test():
    # Score-function derivative of E[(x')^2] under N(theta, sigma^2) is 2 theta.
    torch.manual_seed(13)
    theta=torch.tensor(0.7,requires_grad=True);sigma=0.4
    draw=theta.detach()+sigma*torch.randn(100000)
    dist=torch.distributions.Normal(theta,sigma)
    f=draw.square().detach()
    surrogate=(f*dist.log_prob(draw.detach())).mean()
    grad=torch.autograd.grad(surrogate,theta)[0].item()
    assert abs(grad-1.4)<0.12,(grad,1.4)
    env=CartPole(seed=4);data=collect_data(env,64,6)
    models=train_ensemble(env,data,3,B=2,epochs=1,width=8)
    state=env.reset(); seq=torch.zeros((3,5,1))
    v0,nom,pen=score_candidates(env,models,seq,0.,Q=4,initial_state=state,return_parts=True)
    assert torch.allclose(v0,nom) and torch.equal(pen,torch.zeros_like(pen))
    vr,nomr,pr=score_candidates(env,models,seq,.01,Q=4,initial_state=state,return_parts=True)
    assert torch.isfinite(vr).all() and torch.isfinite(pr).all() and (pr>=0).all()
    assert all(torch.isfinite(p).all() for model in models for p in model.parameters())
    print(f'self-test PASS; Gaussian score Monte Carlo derivative {grad:.3f} vs analytic 1.400; 'f'DR penalties {pr.tolist()}',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',choices=['cartpole','pendulum','both'],default='both')
    ap.add_argument('--out',default='results')
    ap.add_argument('--epsilon',type=float,default=0.002)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--eval-steps',type=int,default=None)
    ap.add_argument('--eval-seeds',type=int,default=None)
    ap.add_argument('--shifts',type=str,default=None,help='Comma-separated physical parameter values, e.g. 0.2,0.5,0.8')
    args=ap.parse_args()
    if args.self_test: return self_test()
    for task in (['cartpole','pendulum'] if args.task=='both' else [args.task]):
        benchmark(task,args.quick,Path(args.out)/task,args.epsilon,args.seed,args.eval_steps,args.eval_seeds,
                  [float(i) for i in args.shifts.split(',')] if args.shifts else None)
if __name__=='__main__':main()
