#!/usr/bin/env python3
"""Post-development DR-PETS experiments, WITHOUT modifying original training/evaluation code.

Audit original saved checkpoints; evaluate 1,000-step bridge; evaluate additional
physics points; regenerate replica-clustered scientific figures. No retraining.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_benchmark import (append_csv, atomic_json, completed_episodes, episode_pool,
                           make_models, sha256_file)

SHARD_SEEDS = {'frankfurt': (101, 103, 107, 109), 'ireland': (113, 127, 131, 137)}
BRIDGE_DEFAULT = {'frankfurt': (101,), 'ireland': (113,)}
BASE = {'nominal': {'mass': 1., 'friction': 1., 'actuation': 1.},
        'low_mass': {'mass': .7, 'friction': 1., 'actuation': 1.},
        'high_mass': {'mass': 1.3, 'friction': 1., 'actuation': 1.},
        'low_friction': {'mass': 1., 'friction': .5, 'actuation': 1.},
        'high_friction': {'mass': 1., 'friction': 1.5, 'actuation': 1.},
        'weak_motor': {'mass': 1., 'friction': 1., 'actuation': .7}}
NEW_GRID = {
    'mass': (.85, 1.15),
    'friction': (.75, 1.25),
    'actuation': (.50, .60, .80, .90),
}
PLOT_GRIDS = {
    'mass': (.70, .85, 1.0, 1.15, 1.30),
    'friction': (.50, .75, 1.0, 1.25, 1.50),
    'actuation': (.50, .60, .70, .80, .90, 1.0),
}
COLUMNS = ['task','shift','seed','method','epsilon','reward','steps','time_sec',
           'forward_displacement','mean_forward_velocity','control_energy']


def fmt_setting(family: str, value: float) -> str:
    return f'{family}_{value:.2f}'.replace('.', 'p')


def make_shift(family: str, value: float):
    return {k: float(value) if k == family else 1. for k in ('mass','friction','actuation')}


def load_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def finite(v):
    return math.isfinite(float(v))


def load_parent(root: Path, seed: int, *, enforce_code=True):
    """Check provenance, 120 unique paired evals, 45 validation rows, and epsilon selection."""
    src = root / f'development_seed_{seed}'
    cp, mf, cf = (src / x for x in ('ensemble.pt', 'run_manifest.json', 'config.json'))
    if not all(p.is_file() for p in (cp, mf, cf, src/'raw.csv', src/'validation.csv')):
        raise RuntimeError(f'Seed {seed} requires ensemble.pt, run_manifest.json, config.json, raw.csv and validation.csv in {src}')
    manifest, config = json.loads(mf.read_text()), json.loads(cf.read_text())
    if manifest.get('schema_version') != 2 or int(manifest.get('settings', {}).get('seed', -1)) != seed:
        raise RuntimeError(f'Seed {seed}: invalid training manifest/seed')
    expected_sha = manifest.get('checkpoint_sha256')
    observed_sha = sha256_file(cp)
    if not expected_sha or observed_sha != expected_sha:
        raise RuntimeError(f'Seed {seed}: checkpoint hash mismatch')
    source_drift = []
    for name, digest in manifest.get('code_sha256', {}).items():
        path = Path(__file__).resolve().parent / name
        if not path.exists() or sha256_file(path) != digest:
            source_drift.append(name)
    if source_drift and enforce_code:
        raise RuntimeError(f'Seed {seed}: original code does not match manifest: {source_drift}. '
                           'Restore the exact original source; do not silently change the trained evaluator.')
    if config.get('seed') != seed or config.get('task') != 'halfcheetah':
        raise RuntimeError(f'Seed {seed}: mismatched config')
    eps = float(config['chosen_epsilon'])
    eval_rows = load_csv(src/'raw.csv')
    val_rows = load_csv(src/'validation.csv')
    expected = {(shift, seed+100003+97*k, method)
                for shift in BASE for k in range(10) for method in ('PETS','DR-PETS-Eq16')}
    observed = [(r['shift'], int(r['seed']), r['method']) for r in eval_rows]
    if len(observed) != 120 or set(observed) != expected or len(set(observed)) != len(observed):
        raise RuntimeError(f'Seed {seed}: incomplete/duplicate original held-out episodes')
    for r in eval_rows:
        if not all(finite(r[k]) for k in ('epsilon','reward','time_sec','forward_displacement',
                                         'mean_forward_velocity','control_energy')) or int(r['steps']) != 150:
            raise RuntimeError(f'Seed {seed}: nonfinite/incomplete evaluation row')
        target = eps if r['method']=='DR-PETS-Eq16' else 0.
        if not math.isclose(float(r['epsilon']), target, abs_tol=1e-10):
            raise RuntimeError(f'Seed {seed}: epsilon does not match selected controller')
    if len(val_rows) != 45:
        raise RuntimeError(f'Seed {seed}: expected exactly 45 validation episodes, found {len(val_rows)}')
    val_keys = [(r['shift'], float(r['epsilon']), int(r['seed'])) for r in val_rows]
    if len(set(val_keys)) != 45 or any(s not in BASE or s=='nominal' or not finite(r['reward']) for r,s in zip(val_rows,[r['shift'] for r in val_rows])):
        raise RuntimeError(f'Seed {seed}: malformed validation rows')
    score = {}
    for e in sorted(set(item[1] for item in val_keys)):
        vals = [np.mean([float(r['reward']) for r in val_rows if r['shift']==sh and float(r['epsilon'])==e])
                for sh in BASE if sh != 'nominal']
        if any(not np.isfinite(x) for x in vals):
            raise RuntimeError(f'Seed {seed}: missing validation condition at epsilon {e}')
        score[e] = float(np.mean(vals)) if config.get('epsilon_selection') == 'mean_shift' else float(min(vals))
    if not math.isclose(eps, max(score, key=score.get), abs_tol=1e-10):
        raise RuntimeError(f'Seed {seed}: chosen epsilon does not match validation selection')
    return {'seed': seed, 'dir': src, 'checkpoint': cp, 'checkpoint_sha256': observed_sha,
            'config': config, 'epsilon': eps, 'validation_scores': score,
            'source_drift': source_drift, 'original_rows': eval_rows}


def check_state_dict(parent, device='cpu'):
    import torch
    states = torch.load(parent['checkpoint'], map_location='cpu', weights_only=True)
    expected = int(parent['config'].get('ensemble', 5))
    if not isinstance(states, (list,tuple)) or len(states) != expected:
        raise RuntimeError(f'Seed {parent["seed"]}: checkpoint member count mismatch')
    model_sha=[]
    for ix, state in enumerate(states):
        if not isinstance(state, dict) or not state:
            raise RuntimeError('Missing state tensors')
        digest=hashlib.sha256()
        for k,v in sorted(state.items()):
            if not torch.is_tensor(v) or not torch.isfinite(v).all():
                raise RuntimeError(f'Seed {parent["seed"]}: nonfinite weights in member {ix}: {k}')
            digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        model_sha.append(digest.hexdigest())
    if len(set(model_sha)) != len(model_sha):
        raise RuntimeError(f'Seed {parent["seed"]}: duplicated ensemble member weights')
    return states, model_sha


def audit_quality(parent, args):
    import torch
    from tasks import make_task
    from run_benchmark import add_data
    torch.set_num_threads(max(1,args.threads))
    seed=parent['seed']
    task=make_task('halfcheetah', BASE['nominal'], seed+900000)
    device=torch.device(args.device)
    try:
        c=parent['config']; cfg=SimpleNamespace(seed=seed, ensemble=c['ensemble'], width=c['width'], depth=c['depth'])
        models=make_models(task,cfg,device)
        states, member_sha=check_state_dict(parent)
        for model, state in zip(models,states): model.load_state_dict(state);model.eval()
        x,actions,nx=add_data(task,args.audit_transitions,seed+900000)
        xt=torch.as_tensor(x,device=device); at=torch.as_tensor(actions,device=device)
        target=torch.as_tensor(nx,device=device); delta=target-xt
        with torch.no_grad():
            mus=[];variances=[];loglike=[]
            for model in models:
                mu,logsd=model(xt,at)
                phys_mu=mu*model.d_std+model.d_mean
                phys_var=logsd.mul(2).exp()*model.d_std.square()
                mus.append(xt+phys_mu)
                variances.append(phys_var)
                ll=torch.distributions.Normal(mu,logsd.exp()).log_prob(
                    (delta-model.d_mean)/model.d_std).sum(-1)-model.d_std.log().sum()
                loglike.append(ll)
            mstack=torch.stack(mus,dim=0)
            pred=mstack.mean(0)
            mixture_mean=pred
            mixture_var=(torch.stack(variances,0)+mstack.square()).mean(0)-mixture_mean.square()
            mix_sd=mixture_var.clamp_min(1e-9).sqrt()
            coverage=((target-mixture_mean).abs()<=1.6448536*mix_sd).float().mean().item()
            mix_ll=torch.logsumexp(torch.stack(loglike,dim=0),0)-math.log(len(models))
            scaled_rmse=((pred-target).square().mean().sqrt()).item()
            no_motion_rmse=((xt-target).square().mean().sqrt()).item()
            # Normalize by holdout distribution, not learned model normalization.
            state_scale=xt.std(0).clamp_min(.1)
            norm_rmse=((((pred-target)/state_scale).square().mean()).sqrt()).item()
            nll=-mix_ll.mean().item()
        horizons=(1,5,15,30)
        error_per_h={h:[] for h in horizons}
        xerr_per_h={h:[] for h in horizons}
        for ep in range(args.audit_trajectories):
            start=task.reset(seed=seed+970000+ep)
            rng=np.random.default_rng(seed+970000+ep)
            actions_seq=[]; observed=[]
            for t in range(30):
                a=task.random_action(rng)
                s,_,done=task.step(a)
                actions_seq.append(a);observed.append(s)
                if done: raise RuntimeError('Unexpected HalfCheetah termination in diagnostic trajectory')
            with torch.no_grad():
                state=torch.as_tensor(start,device=device).float()
                pred_by_model=[state.clone() for _ in models]
                for t,a in enumerate(actions_seq):
                    action=torch.as_tensor(a,dtype=torch.float32,device=device)
                    for j,model in enumerate(models):
                        mu,_=model(pred_by_model[j].unsqueeze(0),action.unsqueeze(0))
                        pred_by_model[j]=pred_by_model[j]+(mu.squeeze(0)*model.d_std+model.d_mean)
                    if t+1 in error_per_h:
                        prediction=torch.stack(pred_by_model).mean(0)
                        actual=torch.as_tensor(observed[t],device=device)
                        error_per_h[t+1].append(float((((prediction-actual)/state_scale).square().mean()).sqrt().item()))
                        xerr_per_h[t+1].append(float(abs(prediction[0]-actual[0]).item()))
        result={
            'seed':seed, 'checkpoint_sha256':parent['checkpoint_sha256'],
            'member_sha256':member_sha, 'model_training_log_available':False,
            'model_training_convergence_verified':False,
            'training_procedure_config':{k:parent['config'].get(k) for k in
                ('train_transitions','rounds','rollouts_per_round','epochs','batch_size','lr')},
            'validation_epsilon':parent['epsilon'], 'validation_scores':{str(k):v for k,v in parent['validation_scores'].items()},
            'heldout_random_transitions':args.audit_transitions,
            'one_step_rmse_raw':scaled_rmse,
            'one_step_no_motion_rmse_raw':no_motion_rmse,
            'one_step_rmse_normalized':norm_rmse,
            'heldout_gaussian_mixture_nll_physical_coordinates':nll,
            'approximate_marginal_90pct_coverage':coverage,
            'multistep_openloop_normalized_rmse':{str(h):float(np.mean(error_per_h[h])) for h in horizons},
            'multistep_openloop_root_x_mae_m':{str(h):float(np.mean(xerr_per_h[h])) for h in horizons},
            'source_drift':parent['source_drift'],
            'interpretation':'Held-out random-action diagnostics, NOT proof of convergence or closed-loop control quality.'
        }
        return result
    finally:task.close()


def manifest_for(parent, name, jobs, c):
    codebase=Path(__file__).resolve().parent
    return {'schema':1, 'experiment':name, 'training_seed':parent['seed'],
            'source_checkpoint_sha256':parent['checkpoint_sha256'],
            'selected_epsilon':parent['epsilon'],
            'settings':{k:v for k,v in vars(c).items() if k not in ('device','workers','worker_threads','threads')},
            'jobs':[{'shift':s,'seed':seed,'method':method,'epsilon':eps,'physics':physics}
                    for s,physics,seed,method,eps in jobs],
            'code_sha256':{p:sha256_file(codebase/p) for p in
                           ('followup.py','run_benchmark.py','eq16_core.py','tasks.py','drpets_benchmark.py')}}


def run_jobs(parent, out: Path, jobs, args, stage: str):
    out.mkdir(parents=True,exist_ok=True)
    cfg=dict(parent['config'])
    cfg.update(task='halfcheetah',seed=parent['seed'],eval_steps=(1000 if stage=='bridge' else 150),
               device=args.device,workers=args.workers,worker_threads=args.worker_threads,
               threads=args.threads,chunk=args.chunk)
    c=SimpleNamespace(**cfg)
    manifest=manifest_for(parent,stage,jobs,c)
    mf=out/'followup_manifest.json'
    if mf.is_file():
        if json.loads(mf.read_text()) != manifest:
            raise RuntimeError(f'Manifest mismatch: {out}. Refusing to mix different experiments.')
    elif (out/'raw.csv').exists():
        raise RuntimeError(f'Orphan raw.csv without followup manifest: {out}')
    else:
        atomic_json(mf,manifest)
    csvpath=out/'raw.csv'
    completed=load_csv(csvpath) if csvpath.is_file() else []
    required={(s, seed,method) for s,physics,seed,method,eps in jobs}
    keys=[(r['shift'],int(r['seed']),r['method']) for r in completed]
    if len(set(keys))!=len(keys) or not set(keys).issubset(required):
        raise RuntimeError(f'{csvpath}: duplicate or unknown jobs')
    for r in completed:
        if not all(finite(r[k]) for k in ('epsilon','reward','time_sec','control_energy','forward_displacement')):
            raise RuntimeError(f'{csvpath}: nonfinite result')
        if not 0<int(r['steps'])<=c.eval_steps:
            raise RuntimeError(f'{csvpath}: invalid episode length')
    done=set(keys)
    todo=[job for job in jobs if (job[0],job[2],job[3]) not in done]
    print(f'[{stage}] seed={parent["seed"]}: completed {len(done)}/{len(jobs)}; running {len(todo)}',flush=True)
    # As-completed saves allow safe interruption. Each job is a complete closed-loop episode.
    if todo:
        import torch
        models=None
        device=torch.device(args.device)
        if args.workers==1:
            from tasks import make_task
            task=make_task('halfcheetah',BASE['nominal'],parent['seed'])
            try:
                models=make_models(task,c,device)
                states,_=check_state_dict(parent)
                for model,state in zip(models,states): model.load_state_dict(state);model.eval()
            finally:task.close()
        with episode_pool(c,parent['checkpoint']) as pool:
            pending={(s,seed,method): (s,physics,seed,method,eps)
                     for s,physics,seed,method,eps in todo}
            # worker requires tuple [task, shiftdict, epsilon, seed, collect, method]
            task_jobs=[('halfcheetah',physics,eps,seed,False,method)
                       for s,physics,seed,method,eps in todo]
            lookup={(tuple(sorted(physics.items())),seed,method):s for s,physics,seed,method,eps in todo}
            for job,res in completed_episodes(task_jobs,pool,models,c,device):
                key=(tuple(sorted(job[1].items())),job[3],job[5])
                shift=lookup[key]
                row={'task':'halfcheetah','shift':shift,'seed':job[3],
                     'method':job[5],'epsilon':job[2],**res}
                append_csv(csvpath,row)
    print(f'[{stage}] seed={parent["seed"]}: complete {len(jobs)}/{len(jobs)}',flush=True)


def report(args):
    import pandas as pd
    from scipy.stats import t as student_t
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    root=Path(args.results_root).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    core_fields=set(COLUMNS)
    for seed in SHARD_SEEDS['frankfurt']+SHARD_SEEDS['ireland']:
        src=root/f'development_seed_{seed}'/'raw.csv'
        if not src.is_file():
            if args.allow_partial: continue
            raise FileNotFoundError(src)
        for r in load_csv(src):
            if not core_fields.issubset(r):raise ValueError(f'Missing columns: {src}')
            all_rows.append({**r,'training_seed':seed,'origin':'original'})
        for family in NEW_GRID:
            ext=root/f'followup_curves_{family}_seed_{seed}'/'raw.csv'
            if ext.exists():
                for r in load_csv(ext):
                    if not core_fields.issubset(r):raise ValueError(f'Missing columns: {ext}')
                    all_rows.append({**r,'training_seed':seed,'origin':'extension'})
    df=pd.DataFrame(all_rows)
    if df.empty:raise RuntimeError('No original data files to plot')
    for field in ('reward','time_sec','mean_forward_velocity','control_energy','forward_displacement','epsilon'):
        df[field]=pd.to_numeric(df[field],errors='raise')
    df['seed']=pd.to_numeric(df['seed'],errors='raise').astype(int)
    duplicated=df.duplicated(['training_seed','shift','seed','method'],keep=False)
    if duplicated.any():raise ValueError('Repeated evaluation job: '+str(df[duplicated][['training_seed','shift','seed','method']].head()))
    if not np.isfinite(df['reward']).all():raise ValueError('Nonfinite rewards')
    df.to_csv(out/'episodes_all.csv',index=False)
    points=[]; pairs=[]
    for (seed,sh),g in df.groupby(['training_seed','shift']):
        p=g[g['method']=='PETS'].set_index('seed')
        d=g[g['method']=='DR-PETS-Eq16'].set_index('seed')
        common=p.index.intersection(d.index)
        for s in common:
            pairs.append({'training_seed':seed,'shift':sh,'seed':s,'delta':float(d.loc[s,'reward']-p.loc[s,'reward'])})
        if len(common)==0:continue
        for method in ['PETS','DR-PETS-Eq16']:
            group=g[g['method']==method]
            points.append({'training_seed':seed,'shift':sh,'method':method,
                           'mean_reward':float(group['reward'].mean()),'n_eval':len(group)})
    points=pd.DataFrame(points);pd.DataFrame(pairs).to_csv(out/'pairs.csv',index=False)
    points.to_csv(out/'per_replica_condition.csv',index=False)
    from collections import OrderedDict
    def summarise(sh):
        rec=[]
        for method in ('PETS','DR-PETS-Eq16'):
            sub=points[(points['shift']==sh)&(points['method']==method)]
            vals=sub['mean_reward'].to_numpy(float)
            if len(vals)==0:continue
            n=len(vals);mean=float(vals.mean())
            half=float(student_t.ppf(.975,n-1)*np.std(vals,ddof=1)/np.sqrt(n)) if n>1 else float('nan')
            rec.append({'shift':sh,'method':method,'n_training_replicas':n,'mean':mean,
                        'ci_low':mean-half,'ci_high':mean+half})
        return rec
    means=pd.DataFrame([r for sh in sorted(points['shift'].unique()) for r in summarise(sh)])
    means.to_csv(out/'per_condition_ci.csv',index=False)
    nominal=means[means['shift']=='nominal']
    def get_plot_name(family,value):
        if value==1.0:return 'nominal'
        if family=='mass':return 'low_mass' if value==.7 else 'high_mass' if value==1.3 else fmt_setting(family,value)
        if family=='friction':return 'low_friction' if value==.5 else 'high_friction' if value==1.5 else fmt_setting(family,value)
        if family=='actuation':return 'weak_motor' if value==.7 else fmt_setting(family,value)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.35),sharey=True,layout='constrained')
    colors={'PETS':'#1766b1','DR-PETS-Eq16':'#c83338'}
    for ix,(family,grid) in enumerate(PLOT_GRIDS.items()):
        ax=axes[ix]
        for method in ('PETS','DR-PETS-Eq16'):
            sub=means[means['method']==method].set_index('shift')
            vals=[(x,sub.loc[get_plot_name(family,x)]) for x in grid if get_plot_name(family,x) in sub.index]
            if not vals:continue
            xs=np.asarray([x for x,r in vals]); ys=np.asarray([r['mean'] for x,r in vals])
            lo=np.asarray([r['ci_low'] for x,r in vals]); hi=np.asarray([r['ci_high'] for x,r in vals])
            ax.plot(xs,ys,'-o' if method=='PETS' else '-s',label='PETS' if method=='PETS' else 'DR-PETS (Eq. 16)',
                    color=colors[method],lw=1.8,markersize=4.5)
            if np.isfinite(lo).all():ax.fill_between(xs,lo,hi,color=colors[method],alpha=.15)
        ax.axvline(1.0,ls=':',lw=.9,color='gray')
        ax.axhline(0,lw=.7,color='gray',alpha=.65)
        ax.set_xticks(grid);ax.set_xticklabels([f'{x:g}' for x in grid],rotation=0)
        ax.set_xlabel({'mass':'Body mass multiplier','friction':'Friction multiplier','actuation':'Actuator strength multiplier'}[family]);
        ax.set_title(f'({"abc"[ix]}) '+{'mass':'Body mass','friction':'Friction','actuation':'Actuator strength'}[family])
        ax.grid(axis='y',alpha=.18)
        if ix==0:ax.set_ylabel('Cumulative reward (150 steps)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1.01),
               ncol=2, frameon=False)
    fig.savefig(out/'physics_curves.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'physics_curves.pdf',bbox_inches='tight')
    fig.savefig(out/'physics_curves.svg',bbox_inches='tight')
    plt.close(fig)
    # Robustness-specific view: performance under shift minus each method's OWN
    # nominal performance. Replicas are paired before computing uncertainty.
    fig, axes = plt.subplots(1,3,figsize=(15.2,4.35),sharey=True,layout='constrained')
    nom_frame=points[points['shift']=='nominal'][['training_seed','method','mean_reward']].rename(
        columns={'mean_reward':'nominal_reward'})
    relative=points.merge(nom_frame,on=['training_seed','method'],how='inner')
    relative['relative_reward']=relative['mean_reward']-relative['nominal_reward']
    relative.to_csv(out/'relative_to_nominal_by_replica.csv',index=False)
    for ix,(family,grid) in enumerate(PLOT_GRIDS.items()):
        ax=axes[ix]
        for method in ('PETS','DR-PETS-Eq16'):
            xs=[];ys=[];los=[];his=[]
            for value in grid:
                key=get_plot_name(family,value)
                vals=relative[(relative['shift']==key)&(relative['method']==method)]['relative_reward'].to_numpy(float)
                if len(vals)<1:continue
                mean=float(vals.mean())
                half=float(student_t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else float('nan')
                xs.append(value);ys.append(mean);los.append(mean-half);his.append(mean+half)
            if xs:
                ax.plot(xs,ys,'-o' if method=='PETS' else '-s',
                        color=colors[method],lw=1.8,markersize=4.5)
                if np.isfinite(los).all():ax.fill_between(xs,los,his,color=colors[method],alpha=.14)
        ax.axhline(0,ls=':',color='gray',lw=1)
        ax.axvline(1.0,ls=':',color='gray',lw=.9)
        ax.set_xticks(grid)
        ax.set_xticklabels([f'{x:g}' for x in grid])
        ax.set_xlabel({'mass':'Body mass multiplier','friction':'Friction multiplier',
                       'actuation':'Actuator strength multiplier'}[family])
        ax.set_title(f'({"abc"[ix]}) '+{'mass':'Body mass','friction':'Friction','actuation':'Actuator strength'}[family])
        ax.grid(axis='y',alpha=.18)
        if ix==0:ax.set_ylabel('Reward minus method-specific nominal return')
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    fig.savefig(out/'relative_to_nominal.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'relative_to_nominal.pdf',bbox_inches='tight')
    plt.close(fig)

    # Summaries of optional 1,000-step bridge and held-out training diagnostics.
    bridge_rows=[]; audit_rows=[]
    for training_seed in SHARD_SEEDS['frankfurt']+SHARD_SEEDS['ireland']:
        bp=root/f'followup_bridge_seed_{training_seed}'/'raw.csv'
        if bp.exists():
            for r in load_csv(bp):bridge_rows.append({**r,'training_seed':training_seed})
        ap=root/f'followup_audit_seed_{training_seed}'/'audit.json'
        if ap.exists():
            record=json.loads(ap.read_text())
            audit_rows.append({k:v for k,v in record.items() if not isinstance(v,(dict,list))})
    if audit_rows:pd.DataFrame(audit_rows).to_csv(out/'model_audit_summary.csv',index=False)
    if bridge_rows:
        bridge=pd.DataFrame(bridge_rows)
        for metric in ('reward','mean_forward_velocity','forward_displacement','control_energy','time_sec'):
            bridge[metric]=pd.to_numeric(bridge[metric],errors='raise')
        bridge['seed']=bridge['seed'].astype(int)
        bridge.to_csv(out/'bridge_episodes.csv',index=False)
        bp=[]
        for (rep,sh),g in bridge.groupby(['training_seed','shift']):
            a=g[g['method']=='PETS'].set_index('seed')
            b=g[g['method']=='DR-PETS-Eq16'].set_index('seed')
            for ss in sorted(a.index.intersection(b.index)):
                bp.append({'training_seed':rep,'shift':sh,'seed':ss,
                           'reward_gain':b.loc[ss,'reward']-a.loc[ss,'reward'],
                           'velocity_gain':b.loc[ss,'mean_forward_velocity']-a.loc[ss,'mean_forward_velocity'],
                           'energy_gain':b.loc[ss,'control_energy']-a.loc[ss,'control_energy']})
        if bp:
            bp=pd.DataFrame(bp);bp.to_csv(out/'bridge_paired.csv',index=False)
            bridge_rep=bp.groupby(['training_seed','shift'])[['reward_gain','velocity_gain','energy_gain']].mean()
            bridge_rep.to_csv(out/'bridge_per_replica_condition.csv')
    # Primary endpoint is computed ONLY from completed original study, not extension points.
    original=df[df['origin']=='original']
    ed=[]; nominal_diffs=[]; shift_diffs=[]
    for seed in sorted(original['training_seed'].unique()):
        means_p=original[(original['training_seed']==seed)&(original['method']=='PETS')].groupby('shift')['reward'].mean()
        means_d=original[(original['training_seed']==seed)&(original['method']=='DR-PETS-Eq16')].groupby('shift')['reward'].mean()
        delta=means_d-means_p
        if not set(BASE).issubset(delta.index):continue
        shift=float(delta[[s for s in BASE if s!='nominal']].mean());nom=float(delta['nominal'])
        ed.append({'training_seed':seed,'shifted_gain':shift,'nominal_gain':nom,'shift_minus_nominal':shift-nom})
        shift_diffs.append(shift);nominal_diffs.append(nom)
    effects=pd.DataFrame(ed);effects.to_csv(out/'replica_primary_effects.csv',index=False)
    summary={'n_training_replicas_primary':len(shift_diffs),
             'mean_shifted_reward_gain':float(np.mean(shift_diffs)) if shift_diffs else None,
             'mean_nominal_reward_gain':float(np.mean(nominal_diffs)) if nominal_diffs else None,
             'note':'Exploratory; no confirmatory claims. Extension points do not create independent training replicas.'}
    if len(shift_diffs)>1:
        ci=float(student_t.ppf(.975,len(shift_diffs)-1)*np.std(shift_diffs,ddof=1)/np.sqrt(len(shift_diffs)))
        summary['primary_95pct_t_ci']=[float(np.mean(shift_diffs)-ci),float(np.mean(shift_diffs)+ci)]
        summary['mean_shift_minus_nominal']=float(effects['shift_minus_nominal'].mean())
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))
    print('Wrote',out/'physics_curves.pdf')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    for action in ('audit','bridge','curves'):
        p=sub.add_parser(action)
        p.add_argument('--shard',choices=['frankfurt','ireland'],required=True)
        p.add_argument('--results-root',default='results')
        p.add_argument('--device',default='cuda')
        p.add_argument('--threads',type=int,default=2)
        p.add_argument('--workers',type=int,default=3)
        p.add_argument('--worker-threads',type=int,default=1)
        p.add_argument('--chunk',type=int,default=16)
        p.add_argument('--seeds',type=int,nargs='*',default=None)
        p.add_argument('--dry-run',action='store_true')
        if action=='audit':
            p.add_argument('--audit-transitions',type=int,default=500)
            p.add_argument('--audit-trajectories',type=int,default=8)
        if action=='curves':p.add_argument('--family',choices=['all',*NEW_GRID],default='all')
        if action=='bridge':p.add_argument('--all-replicas',action='store_true')
    p=sub.add_parser('report')
    p.add_argument('--results-root',default='results')
    p.add_argument('--output',default='results/followup_analysis')
    p.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    if args.stage=='report':return report(args)
    root=Path(args.results_root).resolve()
    seeds=list(args.seeds) if args.seeds is not None and len(args.seeds) else list(
        BRIDGE_DEFAULT[args.shard] if args.stage=='bridge' and not args.all_replicas else SHARD_SEEDS[args.shard])
    if not seeds or any(s not in SHARD_SEEDS[args.shard] for s in seeds): parser.error('Seed(s) not assigned to region')
    if args.stage=='audit' and (args.audit_transitions<10 or args.audit_trajectories<1):parser.error('Invalid audit sample counts')
    if args.stage in ('bridge','curves') and (args.workers<1 or args.chunk<1):parser.error('Invalid worker/chunk count')
    for seed in seeds:
        parent=load_parent(root,seed)
        print(f'Seed {seed}: checkpoint={parent["checkpoint_sha256"][:14]} epsilon={parent["epsilon"]}',flush=True)
        if args.stage=='audit':
            out=root/f'followup_audit_seed_{seed}'/'audit.json'
            if args.dry_run:print('Audit would write',out);continue
            out.parent.mkdir(parents=True,exist_ok=True)
            metrics=audit_quality(parent,args)
            atomic_json(out,metrics)
            print('Audit',seed,metrics['one_step_rmse_raw'],'NLL',metrics['heldout_gaussian_mixture_nll_physical_coordinates'])
        elif args.stage=='bridge':
            conditions=['nominal','low_mass','high_friction','weak_motor']
            jobs=[(s, BASE[s], seed+200003+97*k,method,eps)
                  for s in conditions for k in range(5)
                  for method,eps in (('PETS',0.),('DR-PETS-Eq16',parent['epsilon']))]
            out=root/f'followup_bridge_seed_{seed}'
            if args.dry_run:print('Bridge:',len(jobs),'jobs ->',out);continue
            run_jobs(parent,out,jobs,args,'bridge')
        else:
            families=list(NEW_GRID) if args.family=='all' else [args.family]
            for family in families:
                jobs=[(fmt_setting(family,value),make_shift(family,value),seed+100003+97*k,method,eps)
                      for value in NEW_GRID[family] for k in range(10)
                      for method,eps in (('PETS',0.),('DR-PETS-Eq16',parent['epsilon']))]
                out=root/f'followup_curves_{family}_seed_{seed}'
                if args.dry_run:print('Curves:',family,len(jobs),'jobs ->',out);continue
                run_jobs(parent,out,jobs,args,'curves_'+family)

if __name__=='__main__':main()
