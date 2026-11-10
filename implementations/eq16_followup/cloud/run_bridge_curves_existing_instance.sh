#!/usr/bin/env bash
# One host, one absolute budget; each stage keeps its own status/backup records.
# No provisioning and no shutdown here: the automated launcher owns termination.
set -euo pipefail
: "${SHARD:?Set SHARD=frankfurt or ireland}"
MAX_HOURS="${MAX_HOURS:-24}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
require_positive_integer MAX_HOURS "$MAX_HOURS"
[[ "$SHARD" == frankfurt || "$SHARD" == ireland ]] || { echo 'Invalid SHARD' >&2; exit 2; }
if [[ -z "${DRPETS_DEADLINE_EPOCH:-}" ]]; then
  export DRPETS_DEADLINE_EPOCH="$(($(date +%s)+MAX_HOURS*3600))"
fi
require_positive_integer DRPETS_DEADLINE_EPOCH "$DRPETS_DEADLINE_EPOCH"
echo "Stage queue: shard=$SHARD; representative bridge then curves(all); shared deadline epoch=$DRPETS_DEADLINE_EPOCH"
# The child wrapper's EXIT trap completes backup before the next stage starts.
# A nonzero bridge status stops the queue and never launches curves.
STAGE=bridge BRIDGE_ALL=0 RUN_CURVES_AFTER_BRIDGE=0 \
  bash "$ROOT/cloud/run_followup_existing_instance.sh"
echo 'Bridge completed successfully; starting additional physics curves with remaining budget.'
STAGE=curves FAMILY=all BRIDGE_ALL=0 RUN_CURVES_AFTER_BRIDGE=0 SKIP_INSTALL=1 \
  bash "$ROOT/cloud/run_followup_existing_instance.sh"
echo 'Bridge and all physics-curve families completed successfully.'
