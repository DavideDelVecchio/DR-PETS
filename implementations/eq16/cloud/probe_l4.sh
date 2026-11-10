#!/usr/bin/env bash
# Run on an existing GPU instance after installing project dependencies.
# Uses a trained ensemble; does NOT repeat online training.
set -euo pipefail
ENSEMBLE="${1:-results/cloud/ensemble.pt}"
if [[ ! -f "$ENSEMBLE" ]]; then
  echo "Trained ensemble missing: $ENSEMBLE; pass path to ensemble.pt as argument" >&2
  exit 2
fi
: "${PROBE_STEPS:=5}"
: "${PROBE_EVAL_SEEDS:=1}"
: "${PYTHON:=/opt/pytorch/bin/python}"
[[ -x "$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: $PYTHON" >&2; exit 2; }
"$PYTHON" - <<'PY'
import torch
print('CUDA',torch.cuda.is_available())
if torch.cuda.is_available():
 print('GPU',torch.cuda.get_device_name(0),'VRAM GiB',round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))
PY
mkdir -p results/l4_probes
for spec in '1 8' '1 16' '3 16' '4 16'; do
  read -r WORKERS CHUNK <<< "$spec"
  echo "== workers=${WORKERS} chunk=${CHUNK} $(date -u -Is) =="
  start=$(date +%s)
  "$PYTHON" run_benchmark.py \
    --task halfcheetah --device cuda \
    --ensemble 5 --width 128 --depth 3 \
    --load-ensemble "$ENSEMBLE" \
    --workers "$WORKERS" --worker-threads 1 \
    --eval-seeds "$PROBE_EVAL_SEEDS" --eval-steps "$PROBE_STEPS" \
    --horizon 15 --particles 16 --population 64 --cem-iterations 3 \
    --chunk "$CHUNK" --epsilon .001 --threads 2 \
    --out "results/l4_probes/w${WORKERS}_c${CHUNK}" \
    2>&1 | tee "results/l4_probes/w${WORKERS}_c${CHUNK}.log"
  stop=$(date +%s)
  echo "w${WORKERS} c${CHUNK} elapsed_sec=$((stop-start))" | tee -a results/l4_probes/timings.txt
  nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader || true
done
echo "Compare total elapsed_sec across configurations. GPUs may behave differently from CPU."
