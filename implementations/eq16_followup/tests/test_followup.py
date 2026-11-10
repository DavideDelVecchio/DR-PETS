"""Safety/analysis regression tests for post-study workflows."""
from pathlib import Path
import json
import math
import tempfile
from types import SimpleNamespace
import pytest
import followup


def test_grids_exactly_eight_new_settings():
    assert len([x for xs in followup.NEW_GRID.values() for x in xs]) == 8
    assert len(followup.PLOT_GRIDS['mass']) == 5
    assert len(followup.PLOT_GRIDS['friction']) == 5
    assert len(followup.PLOT_GRIDS['actuation']) == 6
    for family, vals in followup.NEW_GRID.items():
        for val in vals:
            sh=followup.make_shift(family,val)
            assert sh[family] == val
            assert len([key for key in sh if sh[key] != 1.]) == 1
    assert followup.make_shift('actuation', .5)['mass'] == 1.


def test_existing_points_are_not_recomputed_in_extension():
    for family, grid in followup.NEW_GRID.items():
        assert 1.0 not in grid
        if family=='mass': assert not set(grid).intersection({.7,1.3})
        if family=='friction': assert not set(grid).intersection({.5,1.5})
        if family=='actuation': assert .7 not in grid


def test_original_cvs_have_120_unique_paired_jobs():
    root=Path(__file__).resolve().parents[1]/'example_results'
    for seed in (*followup.SHARD_SEEDS['frankfurt'],*followup.SHARD_SEEDS['ireland']):
        rows=followup.load_csv(root/f'development_seed_{seed}'/'raw.csv')
        assert len(rows)==120
        keys=[(r['shift'],r['seed'],r['method']) for r in rows]
        assert len(set(keys))==120
        assert all(int(r['steps'])==150 for r in rows)
        assert all(math.isfinite(float(r['reward'])) for r in rows)


def test_guard_disallows_orphan_csv(tmp_path):
    # Test safety check happens before any GPU/MuJoCo call.
    dst=tmp_path/'stage'
    dst.mkdir()
    (dst/'raw.csv').write_text('task,shift,seed,method\n')
    parent={'seed':101,'epsilon':.001,'checkpoint_sha256':'f'*64,'config':
        {'seed':101,'ensemble':5,'width':128,'depth':3,'task':'halfcheetah'}}
    dummy=SimpleNamespace(device='cpu',workers=1,threads=1,worker_threads=1,chunk=16)
    with pytest.raises(RuntimeError,match='Orphan'):
        followup.run_jobs(parent,dst,[],dummy,'curves_mass')


def test_full_parent_provenance_and_radius_selection(tmp_path):
    import csv
    import hashlib
    import shutil
    import torch
    root=Path(__file__).resolve().parents[1]
    seed=101
    dst=tmp_path/f'development_seed_{seed}'
    dst.mkdir()
    shutil.copy2(root/'example_results'/f'development_seed_{seed}'/'raw.csv',dst/'raw.csv')
    torch.save([{'w':torch.tensor([float(j)])} for j in range(5)],dst/'ensemble.pt')
    val=[]
    for e, reward in [(0.0,2.),(.0001,3.),(.001,5.)]:
        for sh in ['low_mass','high_mass','low_friction','high_friction','weak_motor']:
            for k in range(3):val.append({'shift':sh,'epsilon':e,'seed':seed+700003+107*k,'reward':reward,
                                          'steps':150,'time_sec':1.})
    with (dst/'validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(val[0]));w.writeheader();w.writerows(val)
    (dst/'config.json').write_text(json.dumps({'seed':seed,'task':'halfcheetah','chosen_epsilon':.001,
                'epsilon_selection':'mean_shift','ensemble':5,'width':128,'depth':3}))
    manifest={'schema_version':2,'settings':{'seed':seed},
              'checkpoint_sha256':followup.sha256_file(dst/'ensemble.pt'),
              'code_sha256':{'eq16_core.py':followup.sha256_file(root/'eq16_core.py')}}
    (dst/'run_manifest.json').write_text(json.dumps(manifest))
    result=followup.load_parent(tmp_path,seed)
    assert result['epsilon']==.001
    assert len(result['original_rows'])==120
    assert len(followup.check_state_dict(result)[1])==5
    (dst/'ensemble.pt').write_bytes(b'broken')
    with pytest.raises(RuntimeError,match='checkpoint hash mismatch'):
        followup.load_parent(tmp_path,seed)


def test_stage_job_resume_without_gpu(monkeypatch,tmp_path):
    from contextlib import nullcontext
    src=Path(__file__).resolve().parents[1]
    cp=tmp_path/'ensemble.pt'
    cp.write_bytes(b'fake-for-worker-mocking')
    parent={'seed':101,'epsilon':.001,'checkpoint_sha256':followup.sha256_file(cp),
            'checkpoint':cp,'config':{'task':'halfcheetah','seed':101,'ensemble':5,
                        'width':128,'depth':3,'horizon':15,'particles':16,'population':64,
                        'cem_iterations':3,'gamma':.99,'chunk':16}}
    args=SimpleNamespace(device='cpu',workers=2,worker_threads=1,threads=1,chunk=16)
    monkeypatch.setattr(followup,'episode_pool',lambda *a: nullcontext('no-real-workers'))
    def fake_completed(jobs,*args):
        for job in jobs:
            yield job, {'reward':7.,'steps':150,'time_sec':1.5,'forward_displacement':1.,
                        'mean_forward_velocity':.133333,'control_energy':2.}
    monkeypatch.setattr(followup,'completed_episodes',fake_completed)
    jobs=[(followup.fmt_setting('mass',.85),followup.make_shift('mass',.85),101+100003+97*k,method,eps)
          for k in range(2) for method,eps in [('PETS',0.),('DR-PETS-Eq16',.001)]]
    out=tmp_path/'followup_curves_mass_seed_101'
    followup.run_jobs(parent,out,jobs,args,'curves_mass')
    followup.run_jobs(parent,out,jobs,args,'curves_mass')
    assert len(followup.load_csv(out/'raw.csv'))==4
    with pytest.raises(RuntimeError,match='Manifest mismatch'):
        followup.run_jobs(parent,out,jobs[:-1],args,'curves_mass')
