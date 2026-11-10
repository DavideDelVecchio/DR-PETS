#!/usr/bin/env bash
# Role-based AWS EC2 launcher; safely resumable via periodic S3 sync.
# This DOES provision a billable EC2 instance. Run it only when ready.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
load_deployment_config
: "${S3_BUCKET:?Set S3_BUCKET to your existing S3 bucket}"
: "${INSTANCE_PROFILE:?Set INSTANCE_PROFILE to an existing EC2 S3-access role profile}"
: "${RUN_ID:?Set a stable unique RUN_ID used for resuming, e.g. drpets-eight-20261104}"
: "${SHARD:?Set SHARD=frankfurt|ireland}"
case "$SHARD" in
  frankfurt)
    REGION=eu-central-1; TYPE="${FR_INSTANCE_TYPE:-g6.2xlarge}"
    SUBNET_ID="${FR_SUBNET_ID:-${SUBNET_ID:-}}"
    SECURITY_GROUP_ID="${FR_SECURITY_GROUP_ID:-${SECURITY_GROUP_ID:-}}"
    AMI_ID="${FR_AMI_ID:-${AMI_ID:-}}"
    ;;
  ireland)
    REGION=eu-west-1; TYPE="${IE_INSTANCE_TYPE:-g4dn.2xlarge}"
    SUBNET_ID="${IE_SUBNET_ID:-${SUBNET_ID:-}}"
    SECURITY_GROUP_ID="${IE_SECURITY_GROUP_ID:-${SECURITY_GROUP_ID:-}}"
    AMI_ID="${IE_AMI_ID:-${AMI_ID:-}}"
    ;;
  *) echo 'Invalid SHARD' >&2;exit 2;;
esac
AWS_PROFILE="${AWS_PROFILE:-default}"
STAGE="${STAGE:-development}"
MAX_HOURS="${MAX_HOURS:-24}"
MODE="${MODE:-on-demand}"
WORKERS="${WORKERS:-3}"
[[ "$MODE" == on-demand || "$MODE" == spot ]] || { echo 'Invalid MODE' >&2;exit 2; }
[[ "$MAX_HOURS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_HOURS must be a positive integer' >&2;exit 2; }
[[ "$RUN_ID" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{3,79}$ ]] || { echo 'Invalid RUN_ID' >&2;exit 2; }
[[ "$INSTANCE_PROFILE" =~ ^[a-zA-Z0-9+=,.@_-]+$ ]] || { echo 'Use the instance-profile NAME' >&2; exit 2; }
validate_stage "$STAGE"
require_positive_integer WORKERS "$WORKERS"
PROJECT_NAME="$(basename "$ROOT")"
[[ "$PROJECT_NAME" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo 'Use a project directory without spaces or shell metacharacters' >&2; exit 2; }
resolve_s3_region
TMP="$(mktemp -d)";trap 'rm -rf "$TMP"' EXIT
KEY="drpets-eight/${RUN_ID}/project.zip"
PREFIX="drpets-eight/${RUN_ID}/${SHARD}"
aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$REGION" >/dev/null
# Caller must have quota and capacity; this cannot guarantee either.
(cd "$(dirname "$ROOT")" && zip -qr "$TMP/project.zip" "$(basename "$ROOT")" -x '*/results/*' '*/logs/*' '*/__pycache__/*' '*/.pytest_cache/*')
aws s3 cp "$TMP/project.zip" "s3://$S3_BUCKET/$KEY" --profile "$AWS_PROFILE" --region "$S3_REGION" --only-show-errors
if [[ -z "${AMI_ID:-}" ]]; then
  AMI_ID=$(aws ssm get-parameter --profile "$AWS_PROFILE" --region "$REGION" \
    --name '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id' \
    --query 'Parameter.Value' --output text)
fi
[[ "$AMI_ID" == ami-* ]] || { echo "Could not resolve DLAMI: $AMI_ID" >&2;exit 2; }
ROOT_DEVICE=$(aws ec2 describe-images --profile "$AWS_PROFILE" --region "$REGION" --image-ids "$AMI_ID" --query 'Images[0].RootDeviceName' --output text)
[[ "$ROOT_DEVICE" == /dev/* ]] || { echo 'Bad AMI root device' >&2;exit 2; }
cat > "$TMP/userdata.sh" <<EOF
#!/bin/bash
set -euo pipefail
export AWS_DEFAULT_REGION='$REGION'
export S3_REGION='$S3_REGION'
export S3_BUCKET='$S3_BUCKET'
export PROJECT_KEY='$KEY'
export PREFIX='$PREFIX'
export SHARD='$SHARD'
export STAGE='$STAGE'
export WORKERS='$WORKERS'
export MAX_HOURS='$MAX_HOURS'
mkdir -p /opt/drpets /var/log/drpets
export DRPETS_DEADLINE_EPOCH=\$((\$(date +%s)+MAX_HOURS*3600))
# Independent watchdog; failed bootstrap cannot silently leave EC2 running forever.
# Five minutes beyond the workload budget are reserved for final backup/cleanup.
nohup bash -c "sleep \$((MAX_HOURS*3600+300)); shutdown -h now" >/var/log/drpets/watchdog.log 2>&1 &
set +e
(
  set -euo pipefail
  cd /opt/drpets
  aws s3 cp "s3://\$S3_BUCKET/\$PROJECT_KEY" /tmp/drpets-project.zip --region "\$S3_REGION" --only-show-errors
  unzip -q /tmp/drpets-project.zip
  cd '$PROJECT_NAME'
  # Recover outputs for same run/shard from S3 before resuming.
  # An empty prefix succeeds; an access/transfer error must not start a fresh run.
  aws s3 sync "s3://\$S3_BUCKET/\$PREFIX/results/" results/ --region "\$S3_REGION" --only-show-errors
  export S3_URI="s3://\$S3_BUCKET/\$PREFIX"
  bash cloud/run_on_existing_instance.sh
) >/var/log/drpets/bootstrap.log 2>&1
STATUS=\$?
set -e
printf '%s\n' "DRPETS_EXIT_CODE=\$STATUS" >>/var/log/drpets/bootstrap.log
aws s3 cp /var/log/drpets/bootstrap.log "s3://\$S3_BUCKET/\$PREFIX/logs/bootstrap_\$STAGE.log" --region "\$S3_REGION" --only-show-errors || true
shutdown -h now
EOF
bash -n "$TMP/userdata.sh"
OPTIONS=()
if [[ "$MODE" == spot ]]; then OPTIONS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'); fi
if [[ -n "${SUBNET_ID:-}" ]]; then OPTIONS+=(--subnet-id "$SUBNET_ID"); fi
if [[ -n "${SECURITY_GROUP_ID:-}" ]]; then OPTIONS+=(--security-group-ids "$SECURITY_GROUP_ID"); fi
TAG="drpets-${SHARD}-${RUN_ID}"
INSTANCE_ID=$(aws ec2 run-instances --profile "$AWS_PROFILE" --region "$REGION" \
  --image-id "$AMI_ID" --instance-type "$TYPE" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --user-data "file://$TMP/userdata.sh" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE\",\"Ebs\":{\"VolumeSize\":120,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Experiment,Value=DRPETS8},{Key=RunId,Value=$RUN_ID}]" \
  ${OPTIONS[@]+"${OPTIONS[@]}"} --query 'Instances[0].InstanceId' --output text)
echo "Launched $INSTANCE_ID: $SHARD $REGION $TYPE ($MODE), max ${MAX_HOURS}h; stage=$STAGE"
echo "S3 output prefix: s3://$S3_BUCKET/$PREFIX/"
echo "Re-launch with identical RUN_ID/SHARD after timeout to resume. Check actual EC2 state and billing."
