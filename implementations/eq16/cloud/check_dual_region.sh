#!/usr/bin/env bash
# Read-only preflight for quotas and instance offerings; does not reserve capacity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/aws_common.sh"
load_deployment_config
AWS_PROFILE="${AWS_PROFILE:-default}"
FR_INSTANCE_TYPE="${FR_INSTANCE_TYPE:-g6.2xlarge}"
IE_INSTANCE_TYPE="${IE_INSTANCE_TYPE:-g4dn.2xlarge}"
MODE="${MODE:-on-demand}"
[[ "$MODE" == on-demand || "$MODE" == spot ]] || { echo 'Invalid MODE' >&2; exit 2; }
if [[ "$MODE" == spot ]]; then QUOTA_CODE=L-3819A6DF; else QUOTA_CODE=L-DB2E81BA; fi
LOCAL_PYTHON="${LOCAL_PYTHON:-python3}"
STATUS=0
for x in "eu-central-1:$FR_INSTANCE_TYPE" "eu-west-1:$IE_INSTANCE_TYPE"; do
  region="${x%%:*}"; kind="${x#*:}"
  [[ "$kind" == g* && "$kind" != g4ad.* ]] || { echo 'Use a supported NVIDIA G-family instance type' >&2; exit 2; }
  printf '\n=== %s %s (profile %s) ===\n' "$region" "$kind" "$AWS_PROFILE"
  aws sts get-caller-identity --profile "$AWS_PROFILE" --region "$region" --query '{Account:Account,Arn:Arn}'
  cpu=$(aws ec2 describe-instance-types --profile "$AWS_PROFILE" --region "$region" --instance-types "$kind" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)
  offered=$(aws ec2 describe-instance-type-offerings --profile "$AWS_PROFILE" --region "$region" \
    --location-type region --filters "Name=instance-type,Values=$kind" \
    --query 'length(InstanceTypeOfferings)' --output text)
  quota=$(aws service-quotas get-service-quota --profile "$AWS_PROFILE" --region "$region" \
    --service-code ec2 --quota-code "$QUOTA_CODE" --query 'Quota.Value' --output text)
  echo "type vCPUs: $cpu; regional offering count: $offered; G/VT $MODE vCPU quota: $quota"
  if [[ "$offered" != 1 || "$quota" == None ]]; then echo 'NOT READY: missing offering or quota data'; STATUS=1; continue; fi
  if ! "$LOCAL_PYTHON" - "$cpu" "$quota" <<'PY'
import sys
required=float(sys.argv[1]); available=float(sys.argv[2])
ready=required>0 and available>=required
print('Quota sufficient:',ready)
sys.exit(0 if ready else 1)
PY
  then STATUS=1; fi
  echo 'Note: quota sufficiency does NOT prove free running-instance quota, IAM authorization or live capacity.'
done
exit "$STATUS"
