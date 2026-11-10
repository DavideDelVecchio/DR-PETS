#!/usr/bin/env bash
# Explicitly provisions two follow-up instances. Partial launch remains billable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/aws_common.sh"
source "$ROOT/followup_common.sh"
load_followup_config
STAGE="${STAGE:-audit}"
FAMILY="${FAMILY:-actuation}"
MAX_HOURS="${MAX_HOURS:-2}"
WORKERS="${WORKERS:-3}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
validate_followup_settings
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || { echo 'PREFLIGHT_ONLY must be 0 or 1' >&2; exit 2; }
: "${RUN_ID:?Set original RUN_ID}"
: "${FOLLOWUP_ID:?Set FOLLOWUP_ID}"
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${INSTANCE_PROFILE:?Set INSTANCE_PROFILE}"
for name in SUBNET_ID SECURITY_GROUP_ID AMI_ID S3_RESULTS_URI S3_URI S3_LOGS_URI; do
  [[ -z "${!name:-}" ]] || { echo "Do not use global $name in a dual follow-up launch; configure each region separately or use derived namespaces" >&2; exit 2; }
done
bash "$ROOT/check_dual_region.sh"
for shard in frankfurt ireland; do
  SHARD="$shard" STAGE="$STAGE" FAMILY="$FAMILY" PREFLIGHT_ONLY=1 \
    bash "$ROOT/launch_followup_aws.sh"
done
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo 'Both follow-up shards passed preflight; no instances or source uploads created.'
  exit 0
fi
for shard in frankfurt ireland; do
  echo "=== Launching $shard follow-up ($STAGE) ==="
  SHARD="$shard" STAGE="$STAGE" FAMILY="$FAMILY" MAX_HOURS="$MAX_HOURS" WORKERS="$WORKERS" PREFLIGHT_ONLY=0 \
    bash "$ROOT/launch_followup_aws.sh"
done
