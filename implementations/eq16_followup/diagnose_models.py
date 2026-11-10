#!/usr/bin/env python3
"""Held-out one-step model-fit and random-action baseline quality diagnostics."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from run_benchmark import add_data, make_models, atomic_json
from tasks import make_task

def diagnose(seed,checkpoint,out,device='cuda',n=500):
    task=make_task('halfcheetah',dict(mass=1,friction=1,actuation=1),seed+900000)
    try:
        class Args: pass
        a=Args();a.ensemble=5;a.width=128;a.depth=3;a.seed=seed
        ms=make_models(task,a,torch.device(device))
        states=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if len(states)!=len(ms):raise ValueError('checkpoint member count mismatch')
        for m,state in zip(ms,states):m.load_state_dict(state);m.eval()
        x,u,nxt=add_data(task,n,seed+900000)
        xt=torch.tensor(x,device=device);at=torch.tensor(u,device=device)
        with torch.no_grad():
            predicted=[]
            for m in ms:
                mu,_=m(xt,at)
                predicted.append(xt+mu*m.d_std+m.d_mean)
            pred=torch.stack(predicted).mean(0).cpu().numpy()
        mse=np.square(pred-nxt).mean(axis=0).tolist()
        mae=float(np.abs(pred-nxt).mean())
        random_rewards=[];random_velocities=[]
        for ep in range(5):
            s=task.reset(seed=seed+950000+ep)
            rng=np.random.default_rng(seed+950000+ep)
            ret=0.;first=s[0]
            steps=0
            for t in range(150):
                action=task.random_action(rng)
                s,r,done=task.step(action)
                ret+=float(r);steps+=1
                if done:break
            random_rewards.append(ret)
            random_velocities.append(float(s[0]-first)/(steps*task.dt))
        result={'seed':seed,'heldout_random_transitions':n,'heldout_one_step_mae':mae,
                'heldout_one_step_mse_by_state':mse,
                'random_action_mean_return_150':float(np.mean(random_rewards)),
                'random_action_mean_velocity_150':float(np.mean(random_velocities))}
        atomic_json(out,result)
        print('diagnostics',json.dumps({k:v for k,v in result.items() if k!='heldout_one_step_mse_by_state'}),flush=True)
    finally: task.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--transitions',type=int,default=500)
    a=p.parse_args();diagnose(a.seed,a.checkpoint,Path(a.out),a.device,a.transitions)
if __name__=='__main__':main()
