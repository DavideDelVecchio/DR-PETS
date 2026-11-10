#!/usr/bin/env bash
# Shared helpers only; sourcing this file never creates AWS resources.
load_deployment_config() {
  local config="${DEPLOYMENT_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deployment.env}"
  if [[ -f "$config" ]]; then
    source "$config"
  elif [[ -n "${DEPLOYMENT_CONFIG:-}" ]]; then
    echo "Deployment configuration not found: $config" >&2
    return 2
  fi
}
require_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s must be a positive integer\n' "$1" >&2
    return 2
  }
}

validate_stage() {
  [[ "$1" == development || "$1" == bridge || "$1" == all ]] || {
    printf 'Invalid STAGE: %s (use development, bridge, or all)\n' "$1" >&2
    return 2
  }
}

resolve_s3_region() {
  if [[ -z "${S3_REGION:-}" ]]; then
    local location
    local profile_args=()
    if [[ -n "${AWS_PROFILE:-}" ]]; then profile_args+=(--profile "$AWS_PROFILE"); fi
    location=$(aws s3api get-bucket-location --bucket "$S3_BUCKET" \
      --region "${REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-eu-central-1}}}" \
      ${profile_args[@]+"${profile_args[@]}"} \
      --query LocationConstraint --output text) || return $?
    S3_REGION="$location"
  fi
  case "$S3_REGION" in
    None|null|NULL|'') S3_REGION=us-east-1 ;;
    EU) S3_REGION=eu-west-1 ;;
  esac
  [[ "$S3_REGION" =~ ^[a-z0-9-]+$ ]] || {
    echo 'Could not resolve S3 bucket region; set S3_REGION explicitly' >&2
    return 2
  }
  export S3_REGION
}
