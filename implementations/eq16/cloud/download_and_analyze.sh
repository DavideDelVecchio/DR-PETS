#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
load_deployment_config
: "${S3_BUCKET:?Set S3_BUCKET}"
: "${RUN_ID:?Set RUN_ID}"
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{3,79}$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
AWS_PROFILE="${AWS_PROFILE:-default}"
resolve_s3_region
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || { echo 'Set PYTHON to your local analysis environment interpreter' >&2; exit 2; }
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$ROOT/results/downloads/$RUN_ID}"
for shard in frankfurt ireland; do
  aws s3 sync "s3://$S3_BUCKET/drpets-eight/$RUN_ID/$shard/results/" \
    "$DOWNLOAD_ROOT/" --profile "$AWS_PROFILE" --region "$S3_REGION" --only-show-errors
  echo "Downloaded $shard results"
done
"$PYTHON" "$ROOT/analyze_shards.py" --root "$DOWNLOAD_ROOT" --out "$DOWNLOAD_ROOT/eight_replica_analysis"
