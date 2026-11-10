#!/usr/bin/env bash
# Short EC2 Spot pilot without a custom security group or EC2 IAM instance profile.
# IMPORTANT: requires an authenticated AWS CLI *on the launching machine*.
# Transfers narrowly scoped S3 presigned URLs through EC2 user data.
# Use the role-based cloud/launch_shard_aws.sh for the dual-region study.
set -euo pipefail
: "${S3_BUCKET:?Set S3_BUCKET to an existing bucket in AWS_REGION}"
AWS_REGION="${AWS_REGION:-eu-central-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.2xlarge}"
MODE="${MODE:-on-demand}"
MAX_HOURS="${MAX_HOURS:-2}"
PRESIGN_SECONDS="${PRESIGN_SECONDS:-14400}"
RUN_ARGS="${RUN_ARGS:---task halfcheetah --device cuda --rounds 1 --train-transitions 10000 --workers 3 --worker-threads 1 --chunk 16 --eval-seeds 3 --eval-steps 150 --epsilon-grid 0 .0001 .001 --out results/cloud}"
S3_PREFIX="${S3_PREFIX:-drpets-eq16/$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/cloud/aws_common.sh"
resolve_s3_region
# Optional local checkpoint from a previous EC2 run. Avoid re-training when
# restarting on a new instance. The checkpoint must match architecture args.
if [[ -n "${RESUME_ARCHIVE:-}" ]]; then
  [[ -r "$RESUME_ARCHIVE" ]] || { echo "Cannot read RESUME_ARCHIVE=$RESUME_ARCHIVE" >&2; exit 2; }
  [[ -z "${RESUME_ENSEMBLE:-}" ]] || { echo 'Use RESUME_ARCHIVE or RESUME_ENSEMBLE, not both' >&2; exit 2; }
  RUN_ARGS="$RUN_ARGS --load-ensemble resume/ensemble.pt --resume"
elif [[ -n "${RESUME_ENSEMBLE:-}" ]]; then
  [[ -r "$RESUME_ENSEMBLE" ]] || { echo "Cannot read RESUME_ENSEMBLE=$RESUME_ENSEMBLE" >&2; exit 2; }
  RUN_ARGS="$RUN_ARGS --load-ensemble resume/ensemble.pt"
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
command -v aws >/dev/null || { echo 'AWS CLI is required locally.' >&2; exit 2; }
command -v zip >/dev/null || { echo 'zip is required locally.' >&2; exit 2; }
command -v python3 >/dev/null || { echo 'python3 with boto3 is required locally.' >&2; exit 2; }
[[ "$MAX_HOURS" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_HOURS must be a positive integer' >&2; exit 2; }
[[ "$PRESIGN_SECONDS" =~ ^[0-9]+$ ]] || { echo 'PRESIGN_SECONDS must be an integer' >&2; exit 2; }
(( PRESIGN_SECONDS >= MAX_HOURS * 3600 + 900 && PRESIGN_SECONDS <= 604800 )) || {
  echo 'PRESIGN_SECONDS must cover MAX_HOURS + 15 minutes and be <= 604800' >&2; exit 2;
}
[[ "$MODE" == spot || "$MODE" == on-demand ]] || { echo 'MODE must be spot or on-demand' >&2; exit 2; }
aws sts get-caller-identity --region "$AWS_REGION" >/dev/null
# A default VPC is required when SUBNET_ID is omitted.
if [[ -z "${SUBNET_ID:-}" ]]; then
  DEFAULT_VPC=$(aws ec2 describe-vpcs --region "$AWS_REGION" \
    --filters 'Name=isDefault,Values=true' --query 'Vpcs[0].VpcId' --output text)
  [[ "$DEFAULT_VPC" == vpc-* ]] || { echo 'No default VPC: set SUBNET_ID to a public/NAT subnet' >&2; exit 2; }
fi
# No --security-group-ids: EC2 uses the VPC default security group.
# No --iam-instance-profile: signed S3 URLs replace on-instance AWS API credentials.
(cd "$(dirname "$ROOT")" && zip -qr "$TMP/project.zip" "$(basename "$ROOT")" \
  -x '*/results/*' '*/__pycache__/*' '*/.pytest_cache/*')
if [[ -n "${RESUME_ENSEMBLE:-}" ]]; then
  mkdir -p "$TMP/$(basename "$ROOT")/resume"
  cp "$RESUME_ENSEMBLE" "$TMP/$(basename "$ROOT")/resume/ensemble.pt"
  (cd "$TMP" && zip -q "$TMP/project.zip" "$(basename "$ROOT")/resume/ensemble.pt")
fi
if [[ -n "${RESUME_ARCHIVE:-}" ]]; then
  # Read only three known files from the previous result tarball, without extracting paths.
  RESUME_ARCHIVE="$RESUME_ARCHIVE" DEST="$TMP/$(basename "$ROOT")/resume" python3 - <<'PYRESUME'
import os,tarfile,pathlib,shutil
out=pathlib.Path(os.environ['DEST']); out.mkdir(parents=True,exist_ok=True)
with tarfile.open(os.environ['RESUME_ARCHIVE'],'r:gz') as tf:
    for src,dst,required in [
        ('./results/cloud/ensemble.pt','ensemble.pt',True),
        ('./results/cloud/validation.csv','validation.csv',False),
        ('./results/cloud/raw.csv','raw.csv',False),
    ]:
        try: entry=tf.getmember(src)
        except KeyError:
            if required: raise RuntimeError('Previous result archive missing '+src)
            continue
        if not entry.isfile(): raise RuntimeError('Invalid resume member '+src)
        with tf.extractfile(entry) as stream, (out/dst).open('wb') as f:
            shutil.copyfileobj(stream,f)
PYRESUME
  (cd "$TMP" && zip -qr "$TMP/project.zip" "$(basename "$ROOT")/resume")
fi
# Archive is named according to directory, but cloud-init discovers the extracted root dynamically.
S3_INPUT_KEY="$S3_PREFIX/project.zip"
S3_OUTPUT_KEY="$S3_PREFIX/pilot-results.tar.gz"
aws s3 cp "$TMP/project.zip" "s3://$S3_BUCKET/$S3_INPUT_KEY" --region "$S3_REGION" >/dev/null
# Generate short-lived, object-specific GET and PUT URLs with local AWS credentials.
# IMPORTANT: URLs are embedded in EC2 user data and accessible to users with related EC2 permissions.
PRESIGNED_URLS=$(S3_REGION="$S3_REGION" S3_BUCKET="$S3_BUCKET" \
  S3_INPUT_KEY="$S3_INPUT_KEY" S3_OUTPUT_KEY="$S3_OUTPUT_KEY" \
  PRESIGN_SECONDS="$PRESIGN_SECONDS" python3 - <<'PY'
import boto3, os
from botocore.config import Config
region=os.environ['S3_REGION']
s3=boto3.Session(region_name=region).client(
    's3',
    endpoint_url=f'https://s3.{region}.amazonaws.com',
    config=Config(signature_version='s3v4', s3={'addressing_style': 'virtual'}),
)
sec=int(os.environ['PRESIGN_SECONDS'])
for method,key in [('get_object','S3_INPUT_KEY'), ('put_object','S3_OUTPUT_KEY')]:
    print(s3.generate_presigned_url(method, Params={'Bucket':os.environ['S3_BUCKET'], 'Key':os.environ[key]}, ExpiresIn=sec, HttpMethod='GET' if method=='get_object' else 'PUT'))
PY
)
GET_URL=$(printf '%s\n' "$PRESIGNED_URLS" | sed -n '1p')
PUT_URL=$(printf '%s\n' "$PRESIGNED_URLS" | sed -n '2p')
[[ "$GET_URL" == https://* && "$PUT_URL" == https://* ]] || { echo 'Presigned URL creation failed' >&2; exit 2; }
RUN_ARGS_B64=$(printf '%s' "$RUN_ARGS" | base64 | tr -d '\n')
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
# WARNING: user data contains temporary single-object S3 URLs; do not log or forward it.
set -uo pipefail
export GET_URL='$GET_URL'
export PUT_URL='$PUT_URL'
export RUN_ARGS_B64='$RUN_ARGS_B64'
mkdir -p /opt/drpets /var/log
cat > /opt/drpets/entry.sh <<'INNER'
#!/bin/bash
set -euo pipefail
cd /opt/drpets
curl --fail --location --silent --show-error "\$GET_URL" --output /tmp/drpets-project.zip
mkdir -p /opt/drpets/project
unzip -q /tmp/drpets-project.zip -d /opt/drpets/project
cd /opt/drpets/project/*/
PYTHON=/opt/pytorch/bin/python
[[ -x "\$PYTHON" ]] || { echo "DLAMI PyTorch interpreter missing: \$PYTHON" >&2; exit 2; }
export PATH="/opt/pytorch/bin:\$PATH"
export PYTHONUNBUFFERED=1
"\$PYTHON" -m pip --version
"\$PYTHON" -c 'import sys, torch; print(sys.version); print("PyTorch", torch.__version__)'
"\$PYTHON" -m pip install -r requirements.txt
"\$PYTHON" -m pytest -q tests
"\$PYTHON" cloud/verify_halfcheetah_mass.py
"\$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print(torch.cuda.get_device_name(0))'
read -r -a ARGS <<< "\$(printf '%s' "\$RUN_ARGS_B64" | base64 -d)"
# Restore partial CSVs to the configured output directory before --resume executes.
OUT_DIR=results/cloud
for ((i=0; i<\${#ARGS[@]}; i++)); do
  if [[ "\${ARGS[i]}" == '--out' ]]; then OUT_DIR="\${ARGS[i+1]}"; fi
done
if [[ -d resume ]]; then
  mkdir -p "\$OUT_DIR"
  for file in validation.csv raw.csv; do
    if [[ -f "resume/\$file" ]]; then cp "resume/\$file" "\$OUT_DIR/\$file"; fi
  done
fi
"\$PYTHON" run_benchmark.py "\${ARGS[@]}"
INNER
chmod +x /opt/drpets/entry.sh
# On timeout/failure, still try to upload log + any partial results before terminating.
if timeout --signal=TERM --kill-after=30s '${MAX_HOURS}h' /opt/drpets/entry.sh > /var/log/drpets.log 2>&1; then
  STATUS=0
else
  STATUS=\$?
fi
echo "DRPETS_EXIT_CODE=\$STATUS" >> /var/log/drpets.log
mkdir -p /opt/drpets/export
cp /var/log/drpets.log /opt/drpets/export/drpets.log
a=\$(find /opt/drpets/project -type d -name results -print -quit)
if [[ -n "\$a" ]]; then cp -a "\$a" /opt/drpets/export/results; fi
tar -czf /tmp/drpets-results.tar.gz -C /opt/drpets/export .
if ! curl --fail --silent --show-error --request PUT --upload-file /tmp/drpets-results.tar.gz "\$PUT_URL"; then
  echo 'Results upload failed: verify presigned URL lifetime and S3 PUT permissions.' >> /var/log/drpets.log
fi
shutdown -h now
EOF
OPTIONS=()
if [[ "$MODE" == spot ]]; then
  OPTIONS+=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
if [[ -n "${SUBNET_ID:-}" ]]; then OPTIONS+=(--subnet-id "$SUBNET_ID"); fi
INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --user-data "file://$TMP/userdata.sh" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEVICE_NAME\",\"Ebs\":{\"VolumeSize\":120,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=DRPETS-Eq16-Pilot},{Key=Experiment,Value=Research}]' \
  ${OPTIONS[@]+"${OPTIONS[@]}"} --query 'Instances[0].InstanceId' --output text)
echo "Launched $INSTANCE_ID in $AWS_REGION ($INSTANCE_TYPE, $MODE)"
echo "No custom security group or EC2 IAM instance profile was assigned."
echo "Download results after completion: aws s3 cp s3://$S3_BUCKET/$S3_OUTPUT_KEY ./pilot-results.tar.gz --region $S3_REGION"
echo "IMPORTANT: verify EC2 instance state and billing. AWS CLI credentials are still needed LOCALLY."
echo "IMPORTANT: presigned URLs expire after $PRESIGN_SECONDS seconds, or earlier if your AWS login session expires."
