#!/usr/bin/env python3
"""Combine eight training-replicate effects; do not pseudo-replicate episode seeds."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import t
from run_shard import SPLIT

SHIFTS=['low_mass','high_mass','low_friction','high_friction','weak_motor']

def analyze(root):
    records=[]
    for shard,seeds in SPLIT.items():
        for seed in seeds:
            p=root/f'development_seed_{seed}'
            if not (p/'config.json').exists() or not (p/'run_manifest.json').exists():
                raise ValueError(f'Not complete: {p}')
            with (p/'raw.csv').open(newline='') as f: rows=list(csv.DictReader(f))
            grouped={(r['shift'],int(r['seed']),r['method']):float(r['reward']) for r in rows}
            if len(rows)!=120 or len(grouped)!=120:
                raise ValueError(f'Expected 120 distinct paired evaluation episodes: {p}')
            deltas={}
            for shift in ['nominal']+SHIFTS:
                d=[]
                for k in range(10):
                    evseed=seed+100003+97*k
                    d.append(grouped[(shift,evseed,'DR-PETS-Eq16')]-grouped[(shift,evseed,'PETS')])
                deltas[shift]=float(np.mean(d))
            records.append({'training_seed':seed,'shard':shard,
                            'primary_shift_mean':float(np.mean([deltas[s] for s in SHIFTS])),
                            'nominal_mean':deltas['nominal'],
                            **{f'{s}_mean':deltas[s] for s in SHIFTS},
                            'selected_epsilon':float(json.loads((p/'config.json').read_text())['chosen_epsilon'])})
    d=np.array([r['primary_shift_mean'] for r in records],dtype=float)
    n=len(d); mean=float(np.mean(d)); sd=float(np.std(d,ddof=1)); se=sd/np.sqrt(n)
    crit=float(t.ppf(.975,n-1))
    return records,dict(training_replicates=n,primary_mean_difference=mean,
                        replicate_sd=sd,confidence_interval_95=[mean-crit*se,mean+crit*se],
                        t_statistic=float(mean/se) if se else None,
                        two_sided_p=float(2*t.sf(abs(mean/se),n-1)) if se else None,
                        nominal_mean_difference=float(np.mean([r['nominal_mean'] for r in records])),
                        caveat='Exploratory eight-replicate analysis. Pre-specified confirmatory inference requires separate fresh replicates.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results')
    p.add_argument('--out',default='results/eight_replica_analysis')
    args=p.parse_args()
    records,summary=analyze(Path(args.root))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'replicate_effects.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
