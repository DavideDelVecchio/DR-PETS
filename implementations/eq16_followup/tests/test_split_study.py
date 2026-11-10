"""Small-CPU tests of the split and of safe duplicate/resume behavior."""
from pathlib import Path
from types import SimpleNamespace
import csv, json, os, subprocess, sys
import pytest
from run_shard import SPLIT, BRIDGE, build_dev_command
from run_benchmark import write_or_check_manifest

def test_disjoint_exact_eight_training_seeds():
    assert len(SPLIT['frankfurt'])==4
    assert len(SPLIT['ireland'])==4
    assert set(SPLIT['frankfurt']).isdisjoint(SPLIT['ireland'])
    assert len(set(SPLIT['frankfurt']+SPLIT['ireland']))==8
    assert set(BRIDGE['frankfurt']).issubset(SPLIT['frankfurt'])
    assert set(BRIDGE['ireland']).issubset(SPLIT['ireland'])

def test_resume_manifest_refuses_checkpoint_change(tmp_path):
    params={'seed':101,'task':'pendulum','population':64,'epsilon_grid':[0,0.001],
            'out':str(tmp_path),'device':'cpu','workers':1,'threads':1,
            'worker_threads':1,'resume':False,'load_ensemble':''}
    ckpt=tmp_path/'ensemble.pt'; ckpt.write_bytes(b'A')
    write_or_check_manifest(SimpleNamespace(**params),tmp_path,ckpt,{'nominal':1.0})
    write_or_check_manifest(SimpleNamespace(**params),tmp_path,ckpt,{'nominal':1.0})
    ckpt.write_bytes(b'B')
    with pytest.raises(ValueError,match='manifest mismatch'):
        write_or_check_manifest(SimpleNamespace(**params),tmp_path,ckpt,{'nominal':1.0})

def test_zero_radius_pets_and_drpets_jobs_are_distinct(tmp_path):
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'zero'
    cmd=[sys.executable, 'run_benchmark.py', '--task','pendulum','--smoke',
         '--epsilon-grid','0', '--val-seeds','1', '--workers','2','--worker-threads','1',
         '--threads','1', '--out',str(out)]
    subprocess.run(cmd,cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1'},check=True,stdout=subprocess.DEVNULL)
    with (out/'raw.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==6
    assert len({(r['shift'],r['seed'],r['method']) for r in rows})==6
    assert all(float(r['epsilon'])==0.0 for r in rows)
    # Resume should skip all six without duplicating results.
    subprocess.run(cmd+['--load-ensemble',str(out/'ensemble.pt'),'--resume'],cwd=root,
                   env={**os.environ,'OMP_NUM_THREADS':'1'},check=True,stdout=subprocess.DEVNULL)
    with (out/'raw.csv').open() as f: assert len(list(csv.DictReader(f)))==6

def test_analysis_uses_exactly_eight_training_replicates(tmp_path):
    from analyze_shards import analyze
    from run_shard import SPLIT
    for _region, seeds in SPLIT.items():
        for seed in seeds:
            p=tmp_path/f'development_seed_{seed}';p.mkdir()
            (p/'config.json').write_text(json.dumps({'chosen_epsilon':0.001}))
            (p/'run_manifest.json').write_text('{}')
            with (p/'raw.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['shift','seed','method','reward'])
                writer.writeheader()
                for shift in ('nominal','low_mass','high_mass','low_friction','high_friction','weak_motor'):
                    for k in range(10):
                        ev_seed=seed+100003+97*k
                        writer.writerow({'shift':shift,'seed':ev_seed,'method':'PETS','reward':0.0})
                        writer.writerow({'shift':shift,'seed':ev_seed,'method':'DR-PETS-Eq16','reward':2.0})
    records,summary=analyze(tmp_path)
    assert len(records)==8 and summary['training_replicates']==8
    assert summary['primary_mean_difference']==2.0
