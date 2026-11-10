#!/usr/bin/env bash
# AWS CLI deployment: NO AWS credentials in this script. Run from YOUR authenticated machine.
# REQUIRED: S3_BUCKET, SUBNET_ID, SECURITY_GROUP_ID, INSTANCE_PROFILE (EC2 role S3 read/write).
# Optional: AWS_REGION, INSTANCE_TYPE, MAX_HOURS, MODE, RUN_ARGS, AMI_ID.
set -euo pipefail
: "${S3_BUCKET:?S3_BUCKET is required}"
: "${SUBNET_ID:?SUBNET_ID is required}"
: "${SECURITY_GROUP_ID:?SECURITY_GROUP_ID is required}"
: "${INSTANCE_PROFILE:?INSTANCE_PROFILE is required (IAM instance-profile name)}"
AWS_REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.2xlarge}"
MAX_HOURS="${MAX_HOURS:-12}"
MODE="${MODE:-on-demand}"  # spot or on-demand
RUN_ARGS="${RUN_ARGS:---task halfcheetah --device cuda --rounds 1 --train-transitions 10000 --workers 3 --worker-threads 1 --chunk 16 --eval-seeds 3 --eval-steps 150 --epsilon-grid 0 .0001 .001 --out results/cloud}"
S3_PREFIX="${S3_PREFIX:-drpets-eq16/$(date -u +%Y%m%dT%H%M%SZ)}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "$DIR")"
source "$DIR/cloud/aws_common.sh"
resolve_s3_region
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
command -v aws >/dev/null || { echo 'Install AWS CLI first' >&2; exit 2; }
command -v zip >/dev/null || { echo 'Install zip first' >&2; exit 2; }
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null
(cd "$(dirname "$DIR")" && zip -qr "$TMP/project.zip" "$(basename "$DIR")" -x '*/results/*' '*/__pycache__/*' '*/.pytest_cache/*')
aws s3 cp "$TMP/project.zip" "s3://$S3_BUCKET/$S3_PREFIX/project.zip" --region "$S3_REGION"
if [[ -z "${AMI_ID:-}" ]]; then
  AMI_ID=$(aws ssm get-parameter --region "$AWS_REGION" \
    --name '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.8-ubuntu-24.04/latest/ami-id' \
    --query 'Parameter.Value' --output text)
fi
[[ "$AMI_ID" == ami-* ]] || { echo "Bad AMI_ID: $AMI_ID" >&2; exit 2; }
ROOT_DEVICE_NAME=$(aws ec2 describe-images --region "$AWS_REGION" --image-ids "$AMI_ID" --query 'Images[0].RootDeviceName' --output text)
[[ "$ROOT_DEVICE_NAME" == /dev/* ]] || { echo "Unknown root device: $ROOT_DEVICE_NAME" >&2; exit 2; }
cat > "$TMP/userdata.sh" <<EOF
#!/bin/bash
set -uo pipefail
export AWS_DEFAULT_REGION='$AWS_REGION'
export S3_BUCKET='$S3_BUCKET'
export S3_PREFIX='$S3_PREFIX'
export RUN_ARGS='$RUN_ARGS'
mkdir -p /opt/drpets /var/log
cat > /opt/drpets/entry.sh <<'INNER'
#!/bin/bash
set -euo pipefail
cd /opt/drpets
aws s3 cp 's3://$S3_BUCKET/$S3_PREFIX/project.zip' /tmp/project.zip --region '$S3_REGION'
unzip -q /tmp/project.zip -d /opt/drpets
cd '/opt/drpets/$PROJECT_NAME'
PYTHON=/opt/pytorch/bin/python
[[ -x "\$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: \$PYTHON" >&2; exit 2; }
export PATH="/opt/pytorch/bin:\$PATH"
export PYTHONUNBUFFERED=1
"\$PYTHON" -m pip --version
"\$PYTHON" -c 'import sys, torch; print(sys.version); print("PyTorch", torch.__version__)'
"\$PYTHON" -m pip install -r requirements.txt
"\$PYTHON" -m pytest -q tests
"\$PYTHON" cloud/verify_halfcheetah_mass.py
"\$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available";print(torch.cuda.get_device_name(0))'
"\$PYTHON" run_benchmark.py $RUN_ARGS
INNER
chmod +x /opt/drpets/entry.sh
# Entire workload bounded by MAX_HOURS; results + log copied even on failure/timeout.
timeout --signal=TERM --kill-after=30s '${MAX_HOURS}h' /opt/drpets/entry.sh > /var/log/drpets.log 2>&1
STATUS=\$?
echo "DRPETS_EXIT_CODE=\$STATUS" >> /var/log/drpets.log
aws s3 cp /var/log/drpets.log 's3://$S3_BUCKET/$S3_PREFIX/drpets.log' --region '$S3_REGION' || true
aws s3 sync '/opt/drpets/$PROJECT_NAME/results/' 's3://$S3_BUCKET/$S3_PREFIX/results/' --region '$S3_REGION' || true
shutdown -h now
EOF
# GPU DLAMI has a substantial root volume; on instance shutdown the EC2 instance terminates.
OPTIONS=()
if [[ "$MODE" == spot ]]; then OPTIONS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'); fi
INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --user-data "file://$TMP/userdata.sh" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE_NAME\",\"Ebs\":{\"VolumeSize\":120,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DRPETS-Eq16},{Key=Experiment,Value=Research}]' \
  ${OPTIONS[@]+"${OPTIONS[@]}"} --query 'Instances[0].InstanceId' --output text)
echo "Launched $INSTANCE_ID in $AWS_REGION ($INSTANCE_TYPE, $MODE)"
echo "S3 logs/results: s3://$S3_BUCKET/$S3_PREFIX/"
echo "IMPORTANT: verify instance status and billing. Stop/terminate manually if bootstrap fails."
