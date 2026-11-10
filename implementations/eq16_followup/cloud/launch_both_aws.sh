#!/usr/bin/env bash
# Intentional two-instance launch. AWS permissions, quota, capacity, and EC2 role required.
# If the second launch fails, the first remains billable! Inspect the printed instance ID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/aws_common.sh"
load_deployment_config
: "${RUN_ID:?Choose RUN_ID}"
: "${S3_BUCKET:?Choose S3_BUCKET}"
: "${INSTANCE_PROFILE:?Choose INSTANCE_PROFILE}"
# Subnets, security groups and AMIs cannot be shared across regions.
for name in SUBNET_ID SECURITY_GROUP_ID AMI_ID; do
  [[ -z "${!name:-}" ]] || {
    echo "Use FR_${name} and IE_${name}, not a global $name, when launching both regions" >&2
    exit 2
  }
done
# Keep the 8-replicate development phase separate from the optional long bridge.
STAGE="${STAGE:-development}"
validate_stage "$STAGE"
require_positive_integer MAX_HOURS "${MAX_HOURS:-24}"
require_positive_integer WORKERS "${WORKERS:-3}"
[[ "${MODE:-on-demand}" == on-demand || "${MODE:-on-demand}" == spot ]] || { echo 'Invalid MODE' >&2; exit 2; }
# Fail before creating either instance if either region's quota/offerings fail.
bash "$ROOT/check_dual_region.sh"
echo 'Preflight passed; live capacity may still fail after the first launch.'
for shard in frankfurt ireland; do
  echo "=== Starting $shard shard ==="
  SHARD="$shard" STAGE="$STAGE" bash "$ROOT/launch_shard_aws.sh"
done
