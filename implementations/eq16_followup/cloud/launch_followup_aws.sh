#!/usr/bin/env bash
# Explicit billable follow-up launch; never retrains or starts development.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
source "$ROOT/cloud/followup_common.sh"
load_followup_config
: "${SHARD:?Set SHARD=frankfurt or ireland}"
: "${RUN_ID:?Set the ORIGINAL study RUN_ID}"
: "${FOLLOWUP_ID:?Set a stable unique FOLLOWUP_ID}"
: "${S3_BUCKET:?Set the study bucket}"
: "${INSTANCE_PROFILE:?Set the scoped S3/SSM instance profile name}"
AWS_PROFILE="${AWS_PROFILE:-default}"
STAGE="${STAGE:-audit}"
FAMILY="${FAMILY:-actuation}"
BRIDGE_ALL="${BRIDGE_ALL:-0}"
WORKERS="${WORKERS:-3}"
MAX_HOURS="${MAX_HOURS:-2}"
MODE="${MODE:-on-demand}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
validate_followup_settings
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || { echo 'PREFLIGHT_ONLY must be 0 or 1' >&2; exit 2; }
for name in RUN_ID FOLLOWUP_ID; do
  [[ "${!name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{3,79}$ ]] || { echo "Invalid $name" >&2; exit 2; }
done
[[ "$MODE" == on-demand || "$MODE" == spot ]] || { echo 'Invalid MODE' >&2; exit 2; }
[[ "$INSTANCE_PROFILE" =~ ^[a-zA-Z0-9+=,.@_-]+$ ]] || { echo 'Use an instance-profile NAME' >&2; exit 2; }
BASE_PREFIX="drpets-eight/${RUN_ID}/followup/${FOLLOWUP_ID}"
case "$SHARD" in
  frankfurt)
    REGION=eu-central-1; TYPE="${FR_INSTANCE_TYPE:-g6.2xlarge}"
    SUBNET_ID="${FR_SUBNET_ID:-${SUBNET_ID:-}}"
    SECURITY_GROUP_ID="${FR_SECURITY_GROUP_ID:-${SECURITY_GROUP_ID:-}}"
    AMI_ID="${FR_AMI_ID:-${AMI_ID:-}}"
    S3_RESULTS_URI="${FR_S3_RESULTS_URI:-${S3_RESULTS_URI:-s3://$S3_BUCKET/drpets-eight/$RUN_ID/frankfurt/results}}"
    S3_RESULTS_REGION="${FR_S3_RESULTS_REGION:-${S3_RESULTS_REGION:-}}"
    ;;
  ireland)
    REGION=eu-west-1; TYPE="${IE_INSTANCE_TYPE:-g4dn.2xlarge}"
    SUBNET_ID="${IE_SUBNET_ID:-${SUBNET_ID:-}}"
    SECURITY_GROUP_ID="${IE_SECURITY_GROUP_ID:-${SECURITY_GROUP_ID:-}}"
    AMI_ID="${IE_AMI_ID:-${AMI_ID:-}}"
    S3_RESULTS_URI="${IE_S3_RESULTS_URI:-${S3_RESULTS_URI:-s3://$S3_BUCKET/drpets-eight/$RUN_ID/ireland/results}}"
    S3_RESULTS_REGION="${IE_S3_RESULTS_REGION:-${S3_RESULTS_REGION:-}}"
    ;;
  *) echo 'Invalid SHARD' >&2; exit 2 ;;
esac
S3_URI="${S3_URI:-s3://$S3_BUCKET/$BASE_PREFIX/$SHARD/results}"
S3_LOGS_URI="${S3_LOGS_URI:-s3://$S3_BUCKET/$BASE_PREFIX/$SHARD/logs}"
"$LOCAL_PYTHON" "$ROOT/cloud/followup_paths.py" --source "$S3_RESULTS_URI" \
  --destination "$S3_URI" --logs "$S3_LOGS_URI" --output-bucket "$S3_BUCKET"
resolve_s3_region
resolve_parent_s3_region
PROJECT_NAME="$(basename "$ROOT")"
[[ "$PROJECT_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'Use a project directory without shell metacharacters' >&2; exit 2; }
aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$REGION" >/dev/null
active=$(aws ec2 describe-instances --profile "$AWS_PROFILE" --region "$REGION" \
  --filters "Name=tag:RunId,Values=$RUN_ID" "Name=tag:FollowupId,Values=$FOLLOWUP_ID" \
    "Name=tag:Shard,Values=$SHARD" "Name=instance-state-name,Values=pending,running,stopping,shutting-down" \
  --query 'length(Reservations[].Instances[])' --output text)
[[ "$active" == 0 ]] || { echo 'An active follow-up shard exists; refusing duplicate launch' >&2; exit 3; }
select_followup_seeds
source_bucket="${S3_RESULTS_URI#s3://}"; source_bucket="${source_bucket%%/*}"
source_key="${S3_RESULTS_URI#s3://$source_bucket/}"; source_key="${source_key%/}"
for seed in "${FOLLOWUP_SEEDS[@]}"; do
  for file in ensemble.pt run_manifest.json config.json validation.csv raw.csv; do
    size=$(aws s3api head-object --bucket "$source_bucket" \
      --key "$source_key/development_seed_${seed}/$file" --profile "$AWS_PROFILE" \
      --region "$S3_RESULTS_REGION" --query ContentLength --output text)
    [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "Missing/empty parent artifact for seed $seed: $file" >&2; exit 2; }
  done
done
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "Follow-up preflight passed: $SHARD $STAGE; parent objects present; no source upload or instance launch."
  exit 0
fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
KEY="$BASE_PREFIX/project.zip"
(cd "$(dirname "$ROOT")" && zip -qr "$TMP/project.zip" "$PROJECT_NAME" \
  -x '*/results/*' '*/logs/*' '*/__pycache__/*' '*/.pytest_cache/*' '*/.git/*' \
     '*/.venv*/*' '*/baseline_eight_replica_figures/*' '*/cloud/followup.env' \
     '*/cloud/deployment.env' '*/cloud/iam_policy.local.json')
aws s3 cp "$TMP/project.zip" "s3://$S3_BUCKET/$KEY" --profile "$AWS_PROFILE" --region "$S3_REGION" --only-show-errors
if [[ -z "$AMI_ID" ]]; then
  AMI_ID=$(aws ssm get-parameter --profile "$AWS_PROFILE" --region "$REGION" \
    --name '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id' \
    --query 'Parameter.Value' --output text)
fi
[[ "$AMI_ID" == ami-* ]] || { echo 'Could not resolve the regional DLAMI' >&2; exit 2; }
ROOT_DEVICE=$(aws ec2 describe-images --profile "$AWS_PROFILE" --region "$REGION" \
  --image-ids "$AMI_ID" --query 'Images[0].RootDeviceName' --output text)
[[ "$ROOT_DEVICE" == /dev/* ]] || { echo 'Invalid AMI root device' >&2; exit 2; }
cat >"$TMP/userdata.sh" <<EOF
#!/bin/bash
set -euo pipefail
export AWS_DEFAULT_REGION='$REGION'
export S3_REGION='$S3_REGION'
export S3_RESULTS_REGION='$S3_RESULTS_REGION'
export S3_RESULTS_URI='$S3_RESULTS_URI'
export S3_URI='$S3_URI'
export S3_LOGS_URI='$S3_LOGS_URI'
export SHARD='$SHARD'
export STAGE='$STAGE'
export FAMILY='$FAMILY'
export BRIDGE_ALL='$BRIDGE_ALL'
export WORKERS='$WORKERS'
export MAX_HOURS='$MAX_HOURS'
mkdir -p /opt/drpets /var/log/drpets
export DRPETS_DEADLINE_EPOCH=\$((\$(date +%s)+MAX_HOURS*3600))
nohup bash -c "sleep \$((MAX_HOURS*3600+300)); shutdown -h now" >/var/log/drpets/followup_watchdog.log 2>&1 &
set +e
(
  set -euo pipefail
  cd /opt/drpets
  aws s3 cp 's3://$S3_BUCKET/$KEY' /tmp/drpets-followup.zip --region '$S3_REGION' --only-show-errors
  unzip -q /tmp/drpets-followup.zip
  cd '$PROJECT_NAME'
  bash cloud/run_followup_existing_instance.sh
) >/var/log/drpets/followup_bootstrap.log 2>&1
STATUS=\$?
set -e
printf 'DRPETS_EXIT_CODE=%s\n' "\$STATUS" >>/var/log/drpets/followup_bootstrap.log
aws s3 cp /var/log/drpets/followup_bootstrap.log '${S3_LOGS_URI%/}/bootstrap_${STAGE}_${FAMILY}.log' --region '$S3_REGION' --only-show-errors || true
shutdown -h now
EOF
bash -n "$TMP/userdata.sh"
OPTIONS=()
if [[ "$MODE" == spot ]]; then OPTIONS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'); fi
if [[ -n "$SUBNET_ID" ]]; then OPTIONS+=(--subnet-id "$SUBNET_ID"); fi
if [[ -n "$SECURITY_GROUP_ID" ]]; then OPTIONS+=(--security-group-ids "$SECURITY_GROUP_ID"); fi
INSTANCE_ID=$(aws ec2 run-instances --profile "$AWS_PROFILE" --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$TYPE" --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --user-data "file://$TMP/userdata.sh" --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE\",\"Ebs\":{\"VolumeSize\":120,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=drpets-followup-$SHARD-$FOLLOWUP_ID},{Key=RunId,Value=$RUN_ID},{Key=FollowupId,Value=$FOLLOWUP_ID},{Key=Shard,Value=$SHARD}]" \
  ${OPTIONS[@]+"${OPTIONS[@]}"} --query 'Instances[0].InstanceId' --output text)
echo "Launched $INSTANCE_ID: $SHARD $REGION $TYPE; follow-up=$STAGE family=$FAMILY; max ${MAX_HOURS}h"
echo "Read-only parent: $S3_RESULTS_URI/"
echo "Follow-up output: ${S3_URI%/}/"
echo "Logs: ${S3_LOGS_URI%/}/"
echo 'No development/retraining jobs were requested. Verify actual instance state and billing.'
