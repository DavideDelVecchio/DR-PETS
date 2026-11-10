#!/usr/bin/env bash
# Follow-up only, on an existing GPU node. Does not provision or shut down EC2.
# S3_RESULTS_URI is read-only parent input; S3_URI is separate follow-up output.
set -euo pipefail
: "${SHARD:?Set SHARD=frankfurt or ireland}"
STAGE="${STAGE:-audit}"
FAMILY="${FAMILY:-all}"
WORKERS="${WORKERS:-3}"
MAX_HOURS="${MAX_HOURS:-2}"
BRIDGE_ALL="${BRIDGE_ALL:-0}"
THREADS="${THREADS:-2}"
WORKER_THREADS="${WORKER_THREADS:-1}"
CHUNK="${CHUNK:-16}"
PYTHON="${PYTHON:-/opt/pytorch/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
source "$ROOT/cloud/followup_common.sh"
# Never load the laptop's followup.env/AWS_PROFILE on the EC2 host.
validate_followup_settings
select_followup_seeds
require_positive_integer THREADS "$THREADS"
require_positive_integer WORKER_THREADS "$WORKER_THREADS"
require_positive_integer CHUNK "$CHUNK"
[[ -x "$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: $PYTHON" >&2; exit 2; }
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONUNBUFFERED=1
cd "$ROOT"
START_SECONDS=$SECONDS
mkdir -p results logs
LOG_STEM="followup_${STAGE}_${SHARD}"
if [[ "$STAGE" == curves ]]; then LOG_STEM="followup_curves_${FAMILY}_${SHARD}"; fi
RUN_LOG="logs/${LOG_STEM}.log"
if [[ -n "${S3_RESULTS_URI:-}" || -n "${S3_URI:-}" ]]; then
  : "${S3_RESULTS_URI:?Set the read-only parent results prefix}"
  : "${S3_URI:?Set a different follow-up results prefix}"
  S3_URI="${S3_URI%/}"
  S3_RESULTS_URI="${S3_RESULTS_URI%/}"
  if [[ -z "${S3_LOGS_URI:-}" ]]; then
    if [[ "$S3_URI" == */results ]]; then S3_LOGS_URI="${S3_URI%/results}/logs"
    else S3_LOGS_URI="${S3_URI}-logs"; fi
  fi
  "$PYTHON" cloud/followup_paths.py --source "$S3_RESULTS_URI" --destination "$S3_URI" --logs "$S3_LOGS_URI"
  S3_BUCKET="${S3_URI#s3://}"; S3_BUCKET="${S3_BUCKET%%/*}"
  resolve_s3_region
  resolve_parent_s3_region
fi
SYNC_PID=''
BACKUP_READY=0
sync_outputs() {
  local rc=0
  if [[ "$BACKUP_READY" == 1 ]]; then
    # Frozen development_seed_* inputs are NEVER uploaded to the extension.
    aws s3 sync results/ "$S3_URI/" --region "$S3_REGION" --only-show-errors \
      --exclude '*' --include 'followup_*/*' || rc=$?
  fi
  aws s3 sync logs/ "${S3_LOGS_URI%/}/" --region "$S3_REGION" --only-show-errors || rc=$?
  return "$rc"
}
cleanup() {
  code=$?
  trap - EXIT
  if [[ -n "$SYNC_PID" ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
  fi
  printf 'exit_code=%s\n' "$code" >"logs/${LOG_STEM}_exit.txt"
  if [[ -n "${S3_URI:-}" ]]; then
    if ! sync_outputs; then
      echo 'Final follow-up S3 backup failed; inspect local files and permissions.' >&2
      if [[ "$code" == 0 ]]; then code=1; printf 'exit_code=1\n' >"logs/${LOG_STEM}_exit.txt"; fi
    fi
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
if [[ -n "${S3_RESULTS_URI:-}" ]]; then
  for seed in "${FOLLOWUP_SEEDS[@]}"; do
    aws s3 sync "$S3_RESULTS_URI/development_seed_${seed}/" "results/development_seed_${seed}/" \
      --region "$S3_RESULTS_REGION" --only-show-errors
  done
  # Restore ONLY extension outputs, never overwrite the parent from a resume copy.
  aws s3 sync "$S3_URI/" results/ --region "$S3_REGION" --only-show-errors \
    --exclude '*' --include 'followup_*/*'
fi
BACKUP_READY=1
if [[ -n "${S3_URI:-}" ]]; then
  (
    trap - EXIT TERM INT
    while true; do
      sync_outputs || echo 'Periodic follow-up backup failed; will retry.'
      sleep "${SYNC_INTERVAL:-120}"
    done
  ) >"logs/sync_${LOG_STEM}.log" 2>&1 &
  SYNC_PID=$!
fi
echo "Using DLAMI Python: $PYTHON; follow-up=$STAGE; shard=$SHARD"
"$PYTHON" -m pip --version
"$PYTHON" -c 'import sys, torch; print(sys.version); print("PyTorch:", torch.__version__)'
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  "$PYTHON" -m pip install -r requirements.txt
  "$PYTHON" -m pip uninstall -y pynvml >/dev/null 2>&1 || true
  "$PYTHON" -m pip install nvidia-ml-py
fi
"$PYTHON" -m pytest -q tests
"$PYTHON" cloud/verify_halfcheetah_mass.py
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("GPU:", torch.cuda.get_device_name(0))'
verify=( "$PYTHON" cloud/verify_parent_models.py --shard "$SHARD" --stage "$STAGE" --results-root results )
if [[ "$BRIDGE_ALL" == 1 ]]; then verify+=(--all-replicas); fi
"${verify[@]}"
"$PYTHON" -c 'import importlib.metadata as m,json,platform,pathlib; p={n:m.version(n) for n in ("torch","numpy","gymnasium","mujoco","matplotlib")}; p["platform"]=platform.platform(); pathlib.Path("logs/runtime_environment.json").write_text(json.dumps(p,indent=2))'
cmd=( "$PYTHON" followup.py "$STAGE" --shard "$SHARD" --device cuda \
  --workers "$WORKERS" --worker-threads "$WORKER_THREADS" --threads "$THREADS" \
  --chunk "$CHUNK" --results-root results )
if [[ "$STAGE" == curves ]]; then cmd+=(--family "$FAMILY"); fi
if [[ "$STAGE" == bridge && "$BRIDGE_ALL" == 1 ]]; then cmd+=(--all-replicas); fi
if [[ -n "${DRPETS_DEADLINE_EPOCH:-}" ]]; then
  require_positive_integer DRPETS_DEADLINE_EPOCH "$DRPETS_DEADLINE_EPOCH"
  REMAINING_SECONDS=$((DRPETS_DEADLINE_EPOCH-$(date +%s)))
else
  REMAINING_SECONDS=$((MAX_HOURS*3600-(SECONDS-START_SECONDS)))
fi
(( REMAINING_SECONDS > 0 )) || { echo 'Runtime budget exhausted during setup' >&2; exit 124; }
if timeout --signal=TERM --kill-after=120s "$REMAINING_SECONDS" "${cmd[@]}" >"$RUN_LOG" 2>&1; then
  status=0
else
  status=$?
fi
tail -n 30 "$RUN_LOG" || true
exit "$status"
