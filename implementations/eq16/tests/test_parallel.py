"""Regression test: process-parallel evaluation must not change paired rewards."""
import os
import sys
import subprocess
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]


def _run(out: Path, workers: int, pretrained: Path | None):
    cmd = [sys.executable, 'run_benchmark.py', '--task', 'pendulum', '--smoke',
           '--workers', str(workers), '--worker-threads', '1',
           '--threads', '1', '--out', str(out)]
    if pretrained is not None:
        cmd.extend(['--load-ensemble', str(pretrained)])
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    with (out/'raw.csv').open() as f:
        return [{k: v for k, v in row.items() if k != 'time_sec'}
                for row in csv.DictReader(f)]


def test_serial_and_parallel_episodes_identical(tmp_path):
    serial = tmp_path/'serial'
    parallel = tmp_path/'parallel'
    expected = _run(serial, 1, None)
    actual = _run(parallel, 2, serial/'ensemble.pt')
    key=lambda r:(r['shift'],r['seed'],r['method'],r['epsilon'])
    assert sorted(expected,key=key)==sorted(actual,key=key)
