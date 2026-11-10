"""PETS vs Eq(16) DR-PETS for CartPole/Pendulum/HalfCheetah-v5.

Examples:
  python run_benchmark.py --task pendulum --smoke
  python run_benchmark.py --task halfcheetah --rounds 2 --eval-seeds 20 --device cuda

Both controllers use exactly the same trained Gaussian ensemble and cost function.
This is a research baseline, not a reproduction of published results.
"""
from __future__ import annotations
import argparse, csv, json, time, sys, os, hashlib, tempfile, importlib.metadata
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing as mp
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from eq16_core import GaussianDynamics, RolloutConfig, eq16_scores, train_ensemble
from tasks import make_task


def add_data(task,n,seed,old=None):
    rng=np.random.default_rng(seed)
    current=task.reset(seed=seed)
    xs=[];aa=[];nxs=[]
    for i in range(n):
        if i%200==0:current=task.reset(seed=seed+i)
        a=task.random_action(rng);nxt,_,done=task.step(task.from_planner_action(a))
        xs.append(current);aa.append(a);nxs.append(nxt)
        current=task.reset(seed=seed+i+3000) if done else nxt
    out=tuple(np.asarray(x,dtype=np.float32) for x in (xs,aa,nxs))
    if old is not None:out=tuple(np.concatenate([a,b],axis=0) for a,b in zip(old,out))
    return out


def tensor_data(data,device):return tuple(torch.as_tensor(x,dtype=torch.float32,device=device) for x in data)


def make_models(task,args,device):
    models=[]
    for b in range(args.ensemble):
        torch.manual_seed(args.seed+b*11)
        m=GaussianDynamics(task.obs_dim,task.act_dim,args.width,args.depth).to(device)
        m.act_limit=task.action_limit
        models.append(m)
    return models


def choose_action(task,models,state,epsilon,args,seed,device):
    rng=np.random.default_rng(seed)
    H=args.horizon; N=args.population
    mean=np.full((H,task.act_dim),.5 if task.discrete else 0.,dtype=np.float32)
    std=np.ones_like(mean,dtype=np.float32)*(.3 if task.discrete else .65*task.action_limit)
    elite_count=max(3,int(N*args.elite_fraction))
    x=torch.as_tensor(state,dtype=torch.float32,device=device)
    config=RolloutConfig(horizon=H,particles=args.particles,gamma=args.gamma,chunk=args.chunk)
    best=-np.inf;best_action=None
    for it in range(args.cem_iterations):
        samples=rng.normal(mean,std,size=(N,H,task.act_dim)).astype(np.float32)
        if task.discrete:samples=(samples>.5).astype(np.float32)
        else:samples=np.clip(samples,-task.action_limit,task.action_limit)
        samples[0]=np.round(mean) if task.discrete else mean
        seq=torch.as_tensor(samples,device=device)
        with torch.enable_grad():
            scores,nominal,penalty=eq16_scores(models,x,seq,task.reward_model,config,epsilon,seed=seed+it*1000003)
        vals=scores.detach().cpu().numpy()
        if not np.all(np.isfinite(vals)):raise RuntimeError('non-finite CEM candidate scores')
        ix=int(np.argmax(vals))
        if vals[ix]>best:best=vals[ix];best_action=samples[ix,0].copy()
        elite=samples[np.argsort(vals)[-elite_count:]]
        mean=.25*mean+.75*elite.mean(0)
        std=np.clip(.25*std+.75*elite.std(0),.05,.8*task.action_limit)
    return task.from_planner_action(best_action)


def episode(name,shift,models,eps,args,seed,device,collect=False):
    env=make_task(name,shift,seed)
    try:
        state=env.reset(seed=seed)
        total=0.;traj=[[],[],[]]
        initial_x=float(state[0]) if name=='halfcheetah' else 0.0
        control_energy=0.0
        start=time.monotonic();done=False
        for step in range(args.eval_steps):
            a=choose_action(env,models,state,eps,args,seed*1000007+step*431,device)
            a_vec=np.asarray([a],dtype=np.float32) if env.discrete else np.asarray(a,dtype=np.float32)
            nxt,reward,done=env.step(a)
            if collect:traj[0].append(state);traj[1].append(a_vec);traj[2].append(nxt)
            control_energy += float(np.sum(np.square(a_vec)))
            state=nxt;total+=float(reward)
            if done:break
        displacement=float(state[0])-initial_x if name=='halfcheetah' else 0.0
        average_velocity=displacement/(env.dt*(step+1)) if name=='halfcheetah' else 0.0
        result={'reward':total,'steps':step+1,'time_sec':time.monotonic()-start,
                'forward_displacement':displacement,'mean_forward_velocity':average_velocity,
                'control_energy':control_energy}
        if collect:return result,tuple(np.asarray(x,dtype=np.float32) for x in traj)
        return result
    finally:env.close()



# Independent closed-loop episodes can be evaluated in separate processes.
# Always use spawn for CUDA: forking an initialized CUDA runtime is unsafe.
_WORKER_MODELS = None
_WORKER_ARGS = None
_WORKER_DEVICE = None


def _init_episode_worker(task_name, checkpoint, args_dict, worker_threads):
    global _WORKER_MODELS, _WORKER_ARGS, _WORKER_DEVICE
    torch.set_num_threads(max(1, worker_threads))
    _WORKER_ARGS = SimpleNamespace(**args_dict)
    _WORKER_DEVICE = torch.device(_WORKER_ARGS.device)
    nominal_shift = shifts_for(task_name)['nominal']
    task = make_task(task_name, nominal_shift, _WORKER_ARGS.seed)
    try:
        _WORKER_MODELS = make_models(task, _WORKER_ARGS, _WORKER_DEVICE)
        state_dicts = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if len(state_dicts) != len(_WORKER_MODELS):
            raise ValueError(f'Ensemble checkpoint has {len(state_dicts)} members, expected {len(_WORKER_MODELS)}')
        for model, state in zip(_WORKER_MODELS, state_dicts):
            model.load_state_dict(state)
            model.eval()
    finally:
        task.close()
    print(f'GPU/CPU eval worker {os.getpid()} ready; device={_WORKER_DEVICE}; '
          f'threads={torch.get_num_threads()}', flush=True)


def _run_episode_job(job):
    name, shift, epsilon, seed, collect = job[:5]
    result = episode(name, shift, _WORKER_MODELS, epsilon,
                     _WORKER_ARGS, seed, _WORKER_DEVICE, collect=collect)
    return result


@contextmanager
def episode_pool(args, checkpoint):
    if args.workers <= 1:
        yield None
    else:
        # Separate processes avoid the Python GIL and can submit independent
        # GPU kernels to a single L4. Monitor memory and GPU contention.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_episode_worker,
            initargs=(args.task, str(Path(checkpoint).resolve()), vars(args), args.worker_threads),
        ) as pool:
            yield pool


def completed_episodes(jobs, pool, models, args, device):
    """Yield (job, result) as each episode FINISHES; never block on an earlier slow job."""
    if pool is not None:
        submitted = {pool.submit(_run_episode_job, job): job for job in jobs}
        for future in as_completed(submitted):
            yield submitted[future], future.result()
    else:
        for job in jobs:
            name, shift, epsilon, seed, collect = job[:5]
            yield job, episode(name, shift, models, epsilon, args, seed, device, collect=collect)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.tmp-', suffix='.json', delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv(path, row):
    """Atomic CSV snapshots so S3 sync cannot capture a half-written last row."""
    path=Path(path)
    previous=[]
    if path.exists():
        with path.open(newline='') as f:
            previous=list(csv.DictReader(f))
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=path.parent, prefix='.tmp-', suffix='.csv', delete=False) as f:
        tmp=Path(f.name)
        writer=csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader(); writer.writerows(previous); writer.writerow(row)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_or_check_manifest(args, out, checkpoint, variants):
    """Resume only with the same checkpoint and fully specified experiment design."""
    ignore={'out','load_ensemble','resume','device','workers','worker_threads','threads','smoke'}
    settings={k:v for k,v in vars(args).items() if k not in ignore}
    code_files=['run_benchmark.py','eq16_core.py','tasks.py','drpets_benchmark.py']
    root=Path(__file__).parent
    manifest={
        'schema_version':2,
        'settings':settings,
        'checkpoint_sha256':sha256_file(checkpoint),
        'code_sha256':{name:sha256_file(root/name) for name in code_files},
        'physics_shifts':variants,
    }
    dest=out/'run_manifest.json'
    if dest.exists():
        prior=json.loads(dest.read_text())
        if prior != manifest:
            raise ValueError('Resume manifest mismatch (code/config/checkpoint/physics). '
                             'Use a new --out path. Old checkpoint: '
                             +str(prior.get('checkpoint_sha256'))
                             +' New checkpoint: '+manifest['checkpoint_sha256'])
    elif args.resume and ((out/'validation.csv').exists() or (out/'raw.csv').exists()):
        raise ValueError('Refusing to resume existing CSVs without matching run_manifest.json')
    else:
        atomic_json(dest,manifest)


def read_partial_csv(path, fields):
    """Load completed jobs for --resume; reject duplicate/conflicting entries.

    Files from a previous run are untrusted inputs and are checked against the
    jobs required by the current task/seed/epsilon configuration below.
    """
    if not path.exists():
        return []
    with path.open(newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        if not set(fields).issubset(row):
            raise ValueError(f'Invalid resume CSV row in {path}: {row}')
        row['epsilon']=float(row['epsilon'])
        row['seed']=int(row['seed'])
        row['reward']=float(row['reward'])
        row['steps']=int(row['steps'])
        row['time_sec']=float(row['time_sec'])
        if not np.isfinite(row['reward']):
            raise ValueError(f'Invalid reward in {path}: {row}')
    return rows


def shifts_for(name):
    if name=='halfcheetah':
        return {'nominal':dict(mass=1,friction=1,actuation=1),
                'low_mass':dict(mass=.7,friction=1,actuation=1),
                'high_mass':dict(mass=1.3,friction=1,actuation=1),
                'low_friction':dict(mass=1,friction=.5,actuation=1),
                'high_friction':dict(mass=1,friction=1.5,actuation=1),
                'weak_motor':dict(mass=1,friction=1,actuation=.7)}
    if name=='cartpole':return {'nominal':.5,'short_pole':.2,'long_pole':.8}
    return {'nominal':1.,'light':.5,'heavy':1.5}


def run(args):
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA GPU detected')
    if args.epsilon<0:raise ValueError('epsilon must be nonnegative')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    variants=shifts_for(args.task)
    if args.eval_shifts and not set(args.eval_shifts).issubset(variants):
        raise ValueError('Unknown --eval-shifts: '+str(set(args.eval_shifts)-set(variants)))
    nominal_task=make_task(args.task,variants['nominal'],args.seed)
    print(f'Environment {args.task} state={nominal_task.obs_dim} actions={nominal_task.act_dim} device={device}',flush=True)
    checkpoint=out/'ensemble.pt'
    if args.load_ensemble:
        models=make_models(nominal_task,args,device)
        states=torch.load(args.load_ensemble,map_location='cpu',weights_only=True)
        if len(states)!=len(models):
            raise ValueError(f'Checkpoint contains {len(states)} members; expected {len(models)}')
        for m, state in zip(models,states):
            m.load_state_dict(state)
            m.eval()
        if Path(args.load_ensemble).resolve()!=checkpoint.resolve():
            import shutil
            shutil.copyfile(args.load_ensemble,checkpoint)
        print(f'Loaded trained ensemble from {args.load_ensemble}; skipping data collection and training',flush=True)
    else:
        raw=add_data(nominal_task,args.train_transitions,args.seed)
        models=make_models(nominal_task,args,device)
        t0=time.monotonic()
        train_ensemble(models,tensor_data(raw,device),args.epochs,args.batch_size,args.lr,args.seed)
        print(f'Ensemble trained in {time.monotonic()-t0:.1f}s',flush=True)
        for round_ix in range(args.rounds):
            # All PETS on-policy episodes use an identical frozen model.
            # Safe to parallelize across independent simulation environments.
            torch.save([m.state_dict() for m in models],checkpoint)
            jobs=[(args.task,variants['nominal'],0.,
                   args.seed+6000+round_ix*50+k,True) for k in range(args.rollouts_per_round)]
            with episode_pool(args,checkpoint) as pool:
                # Keep deterministic collection order regardless of worker completion order.
                trajectories={j[3]:traj for j,(_res,traj) in completed_episodes(jobs,pool,models,args,device)}
                extra=[trajectories[j[3]] for j in jobs]
            roll=tuple(np.concatenate([e[i] for e in extra],axis=0) for i in range(3))
            raw=tuple(np.concatenate([v,w],axis=0) for v,w in zip(raw,roll))
            models=make_models(nominal_task,args,device)
            train_ensemble(models,tensor_data(raw,device),args.epochs,args.batch_size,args.lr,args.seed+round_ix+2)
            print(f'Online round {round_ix+1}/{args.rounds}: {len(raw[0])} transitions',flush=True)
        torch.save([m.state_dict() for m in models],checkpoint)

    write_or_check_manifest(args,out,checkpoint,variants)
    # Use a persistent process pool for both validation and final evaluation.
    # Each worker loads the same frozen ensemble ONCE, and independently runs
    # complete control episodes. Unlike CUDA multithreading, CUDA multiprocessing
    # can exploit separate CPU cores while sharing the L4.
    with episode_pool(args,checkpoint) as pool:
        if args.epsilon_grid:
            valrows=[]
            jobs=[]; metadata=[]
            for eps in args.epsilon_grid:
                for name,shift in variants.items():
                    if name=='nominal': continue
                    for k in range(args.val_seeds):
                        seed=args.seed+700003+107*k
                        jobs.append((args.task,shift,eps,seed,False))
                        metadata.append((name,eps,seed))
            valpath=out/'validation.csv'
            if args.resume:
                valrows=read_partial_csv(valpath, {'shift','epsilon','seed','reward','steps','time_sec'})
            elif valpath.exists():
                valpath.unlink()
            required={(name,float(eps),seed) for name,eps,seed in metadata}
            seen=set()
            for row in valrows:
                key=(row['shift'],row['epsilon'],row['seed'])
                if key not in required or key in seen:
                    raise ValueError(f'Resume CSV has an unexpected/duplicate validation job: {key}')
                seen.add(key)
            todo=[(job,meta) for job,meta in zip(jobs,metadata)
                  if (meta[0],float(meta[1]),meta[2]) not in seen]
            print(f'Validation: {len(seen)}/{len(required)} previous jobs, {len(todo)} remaining',flush=True)
            pending={(job[2], job[3], tuple(sorted(job[1].items())) if isinstance(job[1],dict) else job[1]):meta
                     for job,meta in todo}
            for job,res in completed_episodes((job for job,_meta in todo),pool,models,args,device):
                key=(job[2],job[3],tuple(sorted(job[1].items())) if isinstance(job[1],dict) else job[1])
                name,eps,seed=pending[key]
                row=dict(shift=name,epsilon=eps,seed=seed,**res)
                valrows.append(row)
                append_csv(valpath,row)
            if len(valrows)!=len(required):
                raise RuntimeError('Validation incomplete, refusing to select epsilon')
            by_eps={}
            for eps in args.epsilon_grid:
                group=[np.mean([r['reward'] for r in valrows if r['epsilon']==eps and r['shift']==shift])
                       for shift in variants if shift!='nominal']
                by_eps[eps]=float(np.mean(group) if args.epsilon_selection=='mean_shift' else min(group))
            chosen_eps=max(by_eps,key=by_eps.get)
            print(f'Validation {args.epsilon_selection}',by_eps,'chosen epsilon',chosen_eps,flush=True)
        else:
            chosen_eps=args.epsilon

        records=[]; jobs=[]; metadata=[]
        for shift_name,shift in variants.items():
            if args.eval_shifts and shift_name not in args.eval_shifts: continue
            for seed_ix in range(args.eval_seeds):
                seed=args.seed+100003+97*seed_ix
                for method,eps in [('PETS',0.),('DR-PETS-Eq16',chosen_eps)]:
                    jobs.append((args.task,shift,eps,seed,False,method))
                    metadata.append((shift_name,seed_ix,seed,method,eps))
        rawpath=out/'raw.csv'
        if args.resume:
            records=read_partial_csv(rawpath, {'task','shift','seed','method','epsilon','reward','steps','time_sec'})
        elif rawpath.exists():
            rawpath.unlink()
        required={(shift,seed,method,float(eps)) for shift,_i,seed,method,eps in metadata}
        seen=set()
        for row in records:
            key=(row['shift'],row['seed'],row['method'],row['epsilon'])
            if row['task'] != args.task or key not in required or key in seen:
                raise ValueError(f'Resume CSV has unexpected/duplicate evaluation job: {row}')
            seen.add(key)
        todo=[(job,meta) for job,meta in zip(jobs,metadata)
              if (meta[0],meta[2],meta[3],float(meta[4])) not in seen]
        print(f'Held-out: {len(seen)}/{len(required)} previous jobs, {len(todo)} remaining',flush=True)
        pending={(job[5],job[3],tuple(sorted(job[1].items())) if isinstance(job[1],dict) else job[1]):meta
                 for job,meta in todo}
        for job,res in completed_episodes((job for job,_meta in todo),pool,models,args,device):
            key=(job[5],job[3],tuple(sorted(job[1].items())) if isinstance(job[1],dict) else job[1])
            shift_name,seed_ix,seed,method,eps=pending[key]
            record={'task':args.task,'shift':shift_name,'seed':seed,'method':method,
                    'epsilon':eps,**res}
            records.append(record)
            append_csv(rawpath,record)
            if method=='DR-PETS-Eq16':
                print(f'{shift_name}: seed {seed_ix+1}/{args.eval_seeds} done',flush=True)
        if len(records)!=len(required):
            raise RuntimeError('Held-out evaluation incomplete')
    atomic_json(out/'config.json',{**vars(args),'chosen_epsilon':chosen_eps})
    nom=next(iter(variants))
    for shift in variants:
        if args.eval_shifts and shift not in args.eval_shifts: continue
        base=[x for x in records if x['shift']==shift and x['method']=='PETS']
        robust=[x for x in records if x['shift']==shift and x['method']=='DR-PETS-Eq16']
        base_by_seed={x['seed']:x['reward'] for x in base}
        robust_by_seed={x['seed']:x['reward'] for x in robust}
        if set(base_by_seed)!=set(robust_by_seed):
            raise RuntimeError('Paired evaluation missing seeds for '+shift)
        d=np.array([robust_by_seed[seed]-base_by_seed[seed] for seed in sorted(base_by_seed)])
        print(f'{shift:<17} PETS={np.mean([p["reward"] for p in base]):>9.3f}'
              f' DR-PETS={np.mean([r["reward"] for r in robust]):>9.3f}'
              f' paired_delta={d.mean():>9.3f}',flush=True)
    nominal_task.close()


def build_cli():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=['cartpole','pendulum','halfcheetah'],default='pendulum')
    p.add_argument('--out',default='results/eq16')
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--ensemble',type=int,default=5)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--train-transitions',type=int,default=10000)
    p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--rollouts-per-round',type=int,default=5)
    p.add_argument('--eval-seeds',type=int,default=10)
    p.add_argument('--eval-steps',type=int,default=200)
    p.add_argument('--epsilon',type=float,default=.001)
    p.add_argument('--epsilon-grid',type=float,nargs='*',default=[])
    p.add_argument('--val-seeds',type=int,default=3)
    p.add_argument('--epsilon-selection',choices=['mean_shift','maximin'],default='mean_shift',
                   help='How to select epsilon using shifted-condition validation means')
    p.add_argument('--eval-shifts',nargs='*',default=[],
                   help='Evaluate only these shift names; empty means all')
    p.add_argument('--horizon',type=int,default=15)
    p.add_argument('--particles',type=int,default=16)
    p.add_argument('--population',type=int,default=64)
    p.add_argument('--cem-iterations',type=int,default=3)
    p.add_argument('--elite-fraction',type=float,default=.15)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--workers',type=int,default=1,help='Independent episode processes; start with 3 or 4 on g6.2xlarge')
    p.add_argument('--worker-threads',type=int,default=1,help='CPU threads per evaluation process')
    p.add_argument('--load-ensemble',default='',help='Skip training; evaluate a previously saved ensemble.pt')
    p.add_argument('--resume',action='store_true',help='Skip completed validation/held-out jobs in existing CSV files')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:
        for name,value in {'ensemble':2,'width':16,'depth':1,'train_transitions':150,
                 'epochs':2,'batch_size':64,'rounds':0,'eval_seeds':1,
                 'eval_steps':4,'horizon':3,'particles':4,'population':6,'cem_iterations':1,'chunk':3}.items():
            setattr(args,name,value)
    return args

if __name__=='__main__':run(build_cli())
