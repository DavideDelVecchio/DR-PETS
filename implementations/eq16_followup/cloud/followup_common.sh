#!/usr/bin/env bash
# Follow-up helpers: sourcing never launches resources or loads local AWS settings.
load_followup_config() {
  local config="${FOLLOWUP_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/followup.env}"
  if [[ -f "$config" ]]; then
    source "$config"
  elif [[ -n "${FOLLOWUP_CONFIG:-}" ]]; then
    echo "Follow-up configuration not found: $config" >&2
    return 2
  fi
}

validate_followup_settings() {
  [[ "$STAGE" == audit || "$STAGE" == bridge || "$STAGE" == curves ]] || {
    echo 'STAGE must be audit, bridge, or curves (never development)' >&2; return 2;
  }
  [[ "${FAMILY:-all}" == mass || "${FAMILY:-all}" == friction || "${FAMILY:-all}" == actuation || "${FAMILY:-all}" == all ]] || {
    echo 'FAMILY must be mass, friction, actuation, or all' >&2; return 2;
  }
  [[ "${BRIDGE_ALL:-0}" == 0 || "${BRIDGE_ALL:-0}" == 1 ]] || {
    echo 'BRIDGE_ALL must be 0 or 1' >&2; return 2;
  }
  require_positive_integer WORKERS "${WORKERS:-3}"
  require_positive_integer MAX_HOURS "${MAX_HOURS:-2}"
  require_positive_integer SYNC_INTERVAL "${SYNC_INTERVAL:-120}"
}

select_followup_seeds() {
  case "$SHARD" in
    frankfurt) FOLLOWUP_SEEDS=(101 103 107 109) ;;
    ireland) FOLLOWUP_SEEDS=(113 127 131 137) ;;
    *) echo 'SHARD must be frankfurt or ireland' >&2; return 2 ;;
  esac
  if [[ "$STAGE" == bridge && "${BRIDGE_ALL:-0}" == 0 ]]; then
    FOLLOWUP_SEEDS=("${FOLLOWUP_SEEDS[0]}")
  fi
}

resolve_parent_s3_region() {
  if [[ -n "${S3_RESULTS_REGION:-}" ]]; then
    [[ "$S3_RESULTS_REGION" =~ ^[a-z0-9-]+$ ]] || { echo 'Invalid S3_RESULTS_REGION' >&2; return 2; }
    return 0
  fi
  local bucket="${S3_RESULTS_URI#s3://}"
  bucket="${bucket%%/*}"
  if [[ "$bucket" == "${S3_BUCKET:-}" && -n "${S3_REGION:-}" ]]; then
    S3_RESULTS_REGION="$S3_REGION"
  else
    S3_RESULTS_REGION=$(
      S3_BUCKET="$bucket"
      S3_REGION=''
      resolve_s3_region && printf '%s' "$S3_REGION"
    )
  fi
  export S3_RESULTS_REGION
}
