"""Numerical p=2 DR-PETS Equation (16) with likelihood-ratio Equation (21).

Standalone research implementation. Does not use the user's GitHub repository.
The perturbation coordinate is neural-network parameters, measured with Euclidean norm.
Requires a Gaussian transition model. Uses TS-infinity-style fixed-model trajectories,
with Q particles *per ensemble member* rather than the original PETS total-particle budget.
The plug-in norm is biased at finite particle count; not a certified W2 bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
from torch import nn

@dataclass(frozen=True)
class RolloutConfig:
    horizon: int = 15
    particles: int = 16
    gamma: float = 0.99
    chunk: int = 8

class GaussianDynamics(nn.Module):
    """Gaussian delta model with standardized state/action features and targets."""
    def __init__(self, state_dim:int, act_dim:int, width:int=128, depth:int=3):
        super().__init__()
        layers=[];inp=state_dim+act_dim
        for _ in range(depth):
            layers += [nn.Linear(inp,width),nn.SiLU()];inp=width
        layers += [nn.Linear(inp,2*state_dim)]
        self.net=nn.Sequential(*layers)
        self.state_dim=state_dim
        for key,shape,fill in [('x_mean',state_dim,0),('x_std',state_dim,1),
                               ('a_scale',act_dim,1),('d_mean',state_dim,0),('d_std',state_dim,1)]:
            self.register_buffer(key,torch.full((shape,),float(fill)))

    def configure_normalization(self,x:torch.Tensor,a:torch.Tensor,delta:torch.Tensor,action_limit:float):
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0).clamp_min(1e-2))
        self.a_scale.fill_(float(action_limit))
        self.d_mean.copy_(delta.mean(0));self.d_std.copy_(delta.std(0).clamp_min(1e-3))

    def forward(self,x,a):
        features=torch.cat(((x-self.x_mean)/self.x_std,a/self.a_scale),-1)
        z=self.net(features)
        return z[...,:self.state_dim],z[...,self.state_dim:].clamp(-5.0,1.0)

    def sample_and_score(self,x:torch.Tensor,a:torch.Tensor,generator:torch.Generator):
        """Sample p_theta(x'|x,a) and obtain its true score in theta.

        Crucial: stop gradients through the SAMPLE when differentiating log p.
        The physical-delta scaling is parameter-independent and does not change
        the parameter score, hence Normal.log_prob on normalized delta suffices.
        """
        mu,logsd=self(x.detach(),a.detach())
        xi=torch.randn(mu.shape,device=mu.device,dtype=mu.dtype,generator=generator)
        z=mu+logsd.exp()*xi
        delta=z*self.d_std+self.d_mean
        xnext=(x.detach()+delta).detach()
        z_fixed=((xnext-x.detach()-self.d_mean)/self.d_std).detach()
        score_logp=torch.distributions.Normal(mu,logsd.exp()).log_prob(z_fixed).sum(-1)
        return xnext,score_logp


def _score_one_chunk(models,initial,actions,reward_fn:Callable,config:RolloutConfig,epsilon:float,seed:int):
    """Return [M] value, nominal reward and Eq16 penalty.

    r_t may depend on (x_t, a_t, x_{t+1}); reward-to-go includes r_t.
    Scores of future transition densities include all indirect state dependence
    through the likelihood-ratio identity; sampled states are correctly detached.
    """
    M,H,A=actions.shape
    if H!=config.horizon: raise ValueError('horizon mismatch')
    if epsilon and config.particles<2: raise ValueError('LOO baseline needs >=2 particles')
    B=len(models); Q=config.particles;device=initial.device
    discount=(config.gamma**torch.arange(H,device=device,dtype=initial.dtype)).reshape(1,1,-1)
    all_returns=[];all_g2=[]
    for b,model in enumerate(models):
        g=torch.Generator(device=device);g.manual_seed(seed+1009*b)
        state=initial.reshape(1,1,-1).expand(M,Q,-1).clone()
        rewards=[];logp=[]
        for t in range(H):
            a=actions[:,t].unsqueeze(1).expand(M,Q,-1)
            with torch.set_grad_enabled(epsilon>0):
                sn,logpt=model.sample_and_score(state,a,g)
            rewards.append(reward_fn(state,a,sn).detach())
            if epsilon>0:logp.append(logpt)
            state=sn
        discounted=torch.stack(rewards,-1)*discount  # [M,Q,H]
        per_particle_return=discounted.sum(-1)
        all_returns.append(per_particle_return.mean(dim=1))
        if epsilon>0:
            # Eq (21): g_b=E[sum_t (sum_{j>=t} gamma^j r_j) score_t]
            # The current transition's reward r_t MUST be included if r_t=f(x_t,a_t,x_{t+1}).
            r_to_go=torch.flip(torch.cumsum(torch.flip(discounted,[-1]),dim=-1),[-1]).detach()
            baseline=(r_to_go.sum(dim=1,keepdim=True)-r_to_go)/(Q-1)
            adv=(r_to_go-baseline).detach()
            scalar=((adv*torch.stack(logp,-1)).sum(dim=(1,2))/Q)  # one objective/candidate
            # Batched VJP differentiates each candidate independently, without applying any update.
            params=tuple(model.parameters())
            basis=torch.eye(M,device=device,dtype=initial.dtype)
            jac=torch.autograd.grad(scalar,params,grad_outputs=basis,is_grads_batched=True)
            g2=sum(t.reshape(M,-1).square().sum(-1) for t in jac)
            all_g2.append(g2.detach())
    nominal=torch.stack(all_returns).mean(0)
    penalty=torch.stack(all_g2).mean(0).clamp_min(0).sqrt() if epsilon>0 else torch.zeros_like(nominal)
    return (nominal-epsilon*penalty).detach(),nominal.detach(),penalty.detach()


def eq16_scores(models,initial,actions,reward_fn,config:RolloutConfig,epsilon=0.0,seed=1):
    """Compute PETS or Equation (16) score for a batch of action sequences.

    actions [M,H,A], initial [D]. Chunking reduces per-candidate gradient memory.
    When epsilon=0, the same random-stream and ensemble path is used as this
    package's unregularized PETS-style baseline, NOT original PETS source code.
    """
    if not models:raise ValueError('empty ensemble')
    if epsilon<0:raise ValueError('epsilon must be nonnegative')
    results=[[],[],[]]
    for start in range(0,len(actions),config.chunk):
        vals=_score_one_chunk(models,initial,actions[start:start+config.chunk],reward_fn,
                              config,epsilon,seed+104729*start)
        for dest,val in zip(results,vals):dest.append(val)
    return tuple(torch.cat(p) for p in results)


def train_ensemble(models,data,epochs=25,batch_size=256,learning_rate=1e-3,seed=1):
    """Independent bootstrap Gaussian NLL training, shared standardization."""
    x,a,nxt=data
    delta=nxt-x
    N=len(x)
    if N<4:raise ValueError('too few observations')
    device=x.device
    for b,model in enumerate(models):
        model.configure_normalization(x,a,delta,action_limit=float(getattr(model,'act_limit',1.0)))
        rng=torch.Generator(device=device);rng.manual_seed(seed+1009*b)
        idx=torch.randint(0,N,(N,),device=device,generator=rng)
        opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
        model.train()
        for _ in range(epochs):
            perm=idx[torch.randperm(N,device=device,generator=rng)]
            for sel in perm.split(batch_size):
                mu,logsd=model(x[sel],a[sel])
                z=(delta[sel]-model.d_mean)/model.d_std
                nll=(0.5*((z-mu)*torch.exp(-logsd)).square()+logsd).sum(-1).mean()
                opt.zero_grad(set_to_none=True);nll.backward();opt.step()
        model.eval()
    return models
