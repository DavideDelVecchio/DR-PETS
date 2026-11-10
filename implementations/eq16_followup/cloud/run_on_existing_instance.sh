#!/usr/bin/env bash
# Run directly on an existing EC2 instance; does NOT create any AWS resources.
# Example: SHARD=frankfurt STAGE=development bash cloud/run_on_existing_instance.sh
set -euo pipefail
: "${SHARD:?Set SHARD to frankfurt or ireland}"
[[ "$SHARD" == frankfurt || "$SHARD" == ireland ]] || { echo 'Unknown SHARD' >&2; exit 2; }
STAGE="${STAGE:-development}"
WORKERS="${WORKERS:-3}"
MAX_HOURS="${MAX_HOURS:-24}"
PYTHON="${PYTHON:-/opt/pytorch/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
validate_stage "$STAGE"
require_positive_integer WORKERS "$WORKERS"
require_positive_integer MAX_HOURS "$MAX_HOURS"
require_positive_integer SYNC_INTERVAL "${SYNC_INTERVAL:-120}"
[[ -x "$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: $PYTHON" >&2; exit 2; }
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONUNBUFFERED=1
cd "$ROOT"
START_SECONDS=$SECONDS
mkdir -p results logs
LOG_STEM="${SHARD}_${STAGE}"
RUN_LOG="logs/${LOG_STEM}.log"
if [[ -n "${S3_URI:-}" ]]; then
  [[ "$S3_URI" == s3://*/* ]] || { echo 'S3_URI must include a bucket and prefix' >&2; exit 2; }
  S3_BUCKET="${S3_URI#s3://}"; S3_BUCKET="${S3_BUCKET%%/*}"
  resolve_s3_region
fi
sync_outputs() {
  local rc=0
  aws s3 sync results/ "${S3_URI%/}/results/" --region "$S3_REGION" --only-show-errors || rc=$?
  aws s3 sync logs/ "${S3_URI%/}/logs/" --region "$S3_REGION" --only-show-errors || rc=$?
  return "$rc"
}
# Periodic S3 backup requires AWS credentials/instance profile on this EC2 host.
SYNC_PID=''
cleanup() {
  st=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  printf 'exit_code=%s\n' "$st" > "logs/${LOG_STEM}_exit.txt"
  if [[ -n "${S3_URI:-}" ]]; then
    if ! sync_outputs; then
      echo 'Final S3 backup failed; preserve local files and inspect transfer permissions.' >&2
      if [[ "$st" == 0 ]]; then
        st=1
        printf 'exit_code=%s\n' "$st" > "logs/${LOG_STEM}_exit.txt"
      fi
    fi
  fi
  exit "$st"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ -n "${S3_URI:-}" ]]; then
  (
    trap - EXIT TERM INT
    while true; do
      sync_outputs || echo 'Periodic S3 backup failed; will retry.'
      sleep "${SYNC_INTERVAL:-120}"
    done
  ) > "logs/sync_${LOG_STEM}.log" 2>&1 &
  SYNC_PID=$!
fi
echo "Using DLAMI Python: $PYTHON"
"$PYTHON" -m pip --version
"$PYTHON" -c 'import sys, torch; print(sys.version); print("PyTorch:", torch.__version__)'
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -m pip uninstall -y pynvml >/dev/null 2>&1 || true
"$PYTHON" -m pip install nvidia-ml-py
"$PYTHON" -m pytest -q tests
"$PYTHON" cloud/verify_halfcheetah_mass.py
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("CUDA:", torch.cuda.get_device_name(0))'
# Dependency setup and tests count against the same budget as the workload.
if [[ -n "${DRPETS_DEADLINE_EPOCH:-}" ]]; then
  require_positive_integer DRPETS_DEADLINE_EPOCH "$DRPETS_DEADLINE_EPOCH"
  REMAINING_SECONDS=$((DRPETS_DEADLINE_EPOCH-$(date +%s)))
else
  REMAINING_SECONDS=$((MAX_HOURS*3600-(SECONDS-START_SECONDS)))
fi
(( REMAINING_SECONDS > 0 )) || { echo 'Runtime budget exhausted during setup' >&2; exit 124; }
if timeout --signal=TERM --kill-after=120s "$REMAINING_SECONDS" "$PYTHON" run_shard.py \
  --shard "$SHARD" --stage "$STAGE" --device cuda \
  --workers "$WORKERS" --worker-threads 1 --threads 2 --chunk 16 \
  --out-root results >"$RUN_LOG" 2>&1; then
  STATUS=0
else
  STATUS=$?
fi
tail -n 25 "$RUN_LOG" || true
exit "$STATUS"
