#!/usr/bin/env python3
"""Run the development/bridge design on a single GPU node.

Four training replicas are deliberately disjoint across AWS regions.
Independent training+online data are generated per seed. PETS and DR-PETS
within a seed use identical learned dynamics and paired eval reset seeds.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
from run_benchmark import sha256_file

SPLIT = {
    'frankfurt': [101, 103, 107, 109],
    'ireland': [113, 127, 131, 137],
}
BRIDGE = {'frankfurt': [101], 'ireland': [113]}
BRIDGE_SHIFTS=['nominal','low_mass','high_friction','weak_motor']
EPS_GRID=[0.0, 0.0001, 0.001]

def verify_completed_provenance(out, seed):
    manifest_file=out/'run_manifest.json'
    checkpoint=out/'ensemble.pt'
    if not manifest_file.is_file() or not checkpoint.is_file():
        raise RuntimeError(f'Completed CSV without valid manifest/checkpoint: {out}')
    manifest=json.loads(manifest_file.read_text())
    if manifest.get('schema_version')!=2 or manifest.get('settings',{}).get('seed')!=seed:
        raise RuntimeError(f'Mismatched completed-run manifest: {out}')
    if sha256_file(checkpoint)!=manifest.get('checkpoint_sha256'):
        raise RuntimeError(f'Checkpoint has changed for completed seed {seed}')
    src=Path(__file__).resolve().parent
    for name,expected in manifest.get('code_sha256',{}).items():
        if not (src/name).is_file() or sha256_file(src/name)!=expected:
            raise RuntimeError(f'Completed run {seed} has a different code version ({name})')


def build_dev_command(args, seed, out):
    cmd = [sys.executable, 'run_benchmark.py',
        '--task','halfcheetah','--device',args.device,
        '--seed',str(seed),'--out',str(out),
        '--ensemble','5','--width','128','--depth','3',
        '--train-transitions','10000','--epochs','25','--rounds','1',
        '--rollouts-per-round','5',
        '--horizon','15','--particles','16','--population','64',
        '--cem-iterations','3','--chunk',str(args.chunk),
        '--workers',str(args.workers),'--worker-threads',str(args.worker_threads),
        '--threads',str(args.threads),
        '--val-seeds','3','--epsilon-grid',*[str(x) for x in EPS_GRID],
        '--epsilon-selection','mean_shift',
        '--eval-seeds','10','--eval-steps','150',
    ]
    if (out/'run_manifest.json').exists():
        if not (out/'ensemble.pt').is_file():
            raise RuntimeError(f'Manifest exists but checkpoint missing: {out}')
        cmd += ['--load-ensemble',str(out/'ensemble.pt'),'--resume']
    elif any((out/name).exists() for name in ('validation.csv','raw.csv')):
        raise RuntimeError(f'Orphan evaluation files in {out}; refusing unsafe resume')
    return cmd

def build_bridge_command(args, seed, out):
    dev=Path(args.out_root)/f'development_seed_{seed}'
    cfg=dev/'config.json'
    checkpoint=dev/'ensemble.pt'
    if not cfg.exists() or not checkpoint.exists():
        raise RuntimeError(f'Bridge requires completed development for seed {seed}: {dev}')
    epsilon=float(json.loads(cfg.read_text())['chosen_epsilon'])
    cmd=[sys.executable,'run_benchmark.py',
        '--task','halfcheetah','--device',args.device,
        '--seed',str(seed),'--out',str(out),
        '--ensemble','5','--width','128','--depth','3',
        '--train-transitions','10000','--epochs','25','--rounds','1',
        '--rollouts-per-round','5',
        '--horizon','15','--particles','16','--population','64',
        '--cem-iterations','3','--chunk',str(args.chunk),
        '--workers',str(args.workers),'--worker-threads',str(args.worker_threads),
        '--threads',str(args.threads),
        '--epsilon',str(epsilon),'--eval-seeds','5','--eval-steps','1000',
        '--eval-shifts',*BRIDGE_SHIFTS,
        '--load-ensemble',str(checkpoint),
    ]
    if (out/'run_manifest.json').exists():
        cmd += ['--resume']
    elif (out/'raw.csv').exists():
        raise RuntimeError(f'Orphan bridge CSV in {out}; refusing unsafe resume')
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',choices=list(SPLIT),required=True)
    p.add_argument('--stage',choices=['development','bridge','all'],default='development')
    p.add_argument('--device',default='cuda')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--worker-threads',type=int,default=1)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--out-root',default='results')
    p.add_argument('--only-seeds',type=int,nargs='*',default=[])
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.workers<1 or args.chunk<1:
        p.error('workers and chunk must be >= 1')
    root=Path(__file__).resolve().parent
    outroot=Path(args.out_root).resolve()
    args.out_root=str(outroot)
    assigned=SPLIT[args.shard]
    seeds=args.only_seeds or assigned
    if any(s not in assigned for s in seeds):
        p.error('A seed is not assigned to the selected region shard')
    print(f'Shard={args.shard} | assigned seeds={seeds} | stage={args.stage}',flush=True)
    stages=(['development','bridge'] if args.stage=='all' else [args.stage])
    for stage in stages:
        for seed in (seeds if stage=='development' else [s for s in seeds if s in BRIDGE[args.shard]]):
            out=outroot/f'{stage}_seed_{seed}'
            out.mkdir(parents=True,exist_ok=True)
            if stage=='development':
                if (out/'config.json').is_file() and (out/'raw.csv').is_file():
                    # Verify expected number and unique job identities before skipping.
                    import csv
                    with (out/'raw.csv').open(newline='') as f: rows=list(csv.DictReader(f))
                    expected={(shift, seed+100003+97*k, method) for shift in
                        ('nominal','low_mass','high_mass','low_friction','high_friction','weak_motor')
                        for k in range(10) for method in ('PETS','DR-PETS-Eq16')}
                    observed={(r['shift'],int(r['seed']),r['method']) for r in rows}
                    if len(rows)==120 and observed==expected:
                        verify_completed_provenance(out,seed)
                        print(f'[skip] completed development seed={seed}',flush=True)
                        if not args.dry_run and not (out/'model_diagnostics.json').exists():
                            subprocess.run([sys.executable,'diagnose_models.py','--seed',str(seed),
                                            '--checkpoint',str(out/'ensemble.pt'),'--out',
                                            str(out/'model_diagnostics.json'),'--device',args.device],cwd=root,check=True)
                        continue
                cmd=build_dev_command(args,seed,out)
            else:
                if (out/'config.json').is_file() and (out/'raw.csv').is_file():
                    import csv
                    with (out/'raw.csv').open(newline='') as f: rows=list(csv.DictReader(f))
                    expected={(shift,seed+100003+97*k,method) for shift in BRIDGE_SHIFTS
                              for k in range(5) for method in ('PETS','DR-PETS-Eq16')}
                    observed={(r['shift'],int(r['seed']),r['method']) for r in rows}
                    if len(rows)==40 and observed==expected:
                        verify_completed_provenance(out,seed)
                        print(f'[skip] completed bridge seed={seed}',flush=True)
                        continue
                cmd=build_bridge_command(args,seed,out)
            print('[run]', ' '.join(cmd),flush=True)
            if not args.dry_run:
                subprocess.run(cmd,cwd=root,check=True)
                if stage=='development' and not (out/'model_diagnostics.json').exists():
                    subprocess.run([sys.executable,'diagnose_models.py','--seed',str(seed),
                                    '--checkpoint',str(out/'ensemble.pt'),
                                    '--out',str(out/'model_diagnostics.json'),'--device',args.device],cwd=root,check=True)
    print('Shard completed successfully.',flush=True)

if __name__=='__main__':main()
