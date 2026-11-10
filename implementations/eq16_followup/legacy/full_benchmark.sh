#!/usr/bin/env bash
# Run from project root on a CUDA instance, after dependencies installed.
# Each training seed is independent. Results are written after every episode.
set -euo pipefail
: "${SEEDS:=101 103 107 109 113}"
: "${PYTHON:=/opt/pytorch/bin/python}"
[[ -x "$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: $PYTHON" >&2; exit 2; }
for seed in $SEEDS; do
  "$PYTHON" run_benchmark.py \
    --task halfcheetah --device cuda --seed "$seed" \
    --ensemble 5 --width 128 --depth 3 \
    --train-transitions 20000 --epochs 35 --rounds 3 --rollouts-per-round 5 \
    --horizon 20 --particles 24 --population 160 --cem-iterations 4 --chunk 16 --workers 3 --worker-threads 1 \
    --val-seeds 4 --epsilon-grid 0 0.00001 0.0001 0.001 0.01 \
    --eval-seeds 20 --eval-steps 1000 --out "results/full_seed_${seed}"
done
