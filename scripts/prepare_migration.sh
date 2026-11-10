#!/usr/bin/env bash
# Preserve history and stage a reviewed replacement in an isolated clean clone.
# Defaults: no commits, no pushes, no master changes, no legacy README import.
set -euo pipefail
export GIT_PAGER=cat
export GIT_TERMINAL_PROMPT=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${MIGRATION_BUNDLE:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO="${1:-$PWD}"
EXPECTED="${EXPECTED_MASTER_SHA:-a4920349d32aec8a4c7f360049979c4848a31239}"
EXPECTED_REMOTE="${EXPECTED_REMOTE:-https://github.com/DavideDelVecchio/DR-PETS.git}"
LOCAL_ONLY="${LOCAL_ONLY:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-1}"
EXCLUDE_LEGACY_README="${EXCLUDE_LEGACY_README:-1}"
LEGACY="legacy/state-action-density-2025"
RESEARCH="research/eq16-benchmarks"
TAG="legacy-density-score-v0.1.0"
for flag in "$LOCAL_ONLY" "$PREPARE_ONLY"; do
  [[ "$flag" == 0 || "$flag" == 1 ]] || { echo 'Mode flags must be 0 or 1' >&2; exit 2; }
done
[[ "$EXCLUDE_LEGACY_README" == 1 ]] || { echo 'This migration must exclude README_LEGACY.md' >&2; exit 2; }
[[ "$EXPECTED" =~ ^[a-fA-F0-9]{40}$ ]] || { echo 'Invalid expected master SHA' >&2; exit 2; }
[[ "$LOCAL_ONLY" == 1 || "$PREPARE_ONLY" == 0 ]] || { echo 'Publishing requires PREPARE_ONLY=0 explicitly' >&2; exit 2; }
[[ -d "$BUNDLE/new_master" ]] || {
  echo 'Migration payload missing; set MIGRATION_BUNDLE to the original migration package.' >&2; exit 2;
}
cd "$REPO"
command -v git >/dev/null
[[ "$(git rev-parse --is-inside-work-tree)" == true ]] || { echo 'Run in a clean Git working tree.' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo 'Working tree not clean; commit or stash changes.' >&2; exit 2; }
REMOTE="$(git remote get-url origin)"
NORMALIZED="${REMOTE/git@github.com:/https://github.com/}"
[[ "${NORMALIZED%.git}" == "${EXPECTED_REMOTE%.git}" ]] || { echo "Unexpected remote: $REMOTE" >&2; exit 2; }
[[ -n "$(git config user.name || true)" && -n "$(git config user.email || true)" ]] || {
 echo 'Configure git user.name and user.email before running.' >&2; exit 2;
}
git fetch origin --tags
BASE="$(git rev-parse refs/remotes/origin/master)"
if [[ "$BASE" != "$EXPECTED" ]]; then
  echo "The remote master has moved from the audited snapshot. REFUSING automatic migration." >&2
  echo "Expected $EXPECTED, observed $BASE; review the diff and update this script intentionally." >&2
  exit 3
fi
if git show-ref --verify --quiet "refs/heads/$LEGACY" || git show-ref --verify --quiet "refs/remotes/origin/$LEGACY"; then
  echo "Legacy branch already exists: $LEGACY. Aborting to avoid rewriting it." >&2; exit 3
fi
if git show-ref --verify --quiet "refs/heads/$RESEARCH" || git show-ref --verify --quiet "refs/remotes/origin/$RESEARCH"; then
  echo "Research branch already exists: $RESEARCH. Aborting to avoid rewriting it." >&2; exit 3
fi
if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
 echo "Historical tag already exists: $TAG. Aborting to avoid duplicate tags." >&2; exit 3
fi
# Preserve the original tree exactly. Do not commit the excluded README under
# another name; classification belongs in the modern implementation docs.
git branch "$LEGACY" "$BASE"
git tag -a "$TAG" "$BASE" -m 'Original 2025 state-action density-score prototype, not Eq16'
# Build replacement on a separate non-destructive branch.
git checkout -b "$RESEARCH" "$BASE"
git rm -r --quiet -- .
# Exclude private/generated files even if accidentally present in the payload.
tar -C "$BUNDLE/new_master" \
  --exclude=.git --exclude=README_LEGACY.md --exclude=.DS_Store \
  --exclude=results --exclude=logs --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude='.venv*' --exclude=.env \
  --exclude=deployment.env --exclude=iam_policy.local.json \
  --exclude=followup.env --exclude=baseline_eight_replica_figures \
  --exclude=iam_policy.drpets-eight.json --exclude='*.pem' --exclude='*.key' \
  -cf - . | tar -xf -
git add -A
while IFS= read -r -d '' file; do
  case "$file" in
    README_LEGACY.md|*/README_LEGACY.md)
      echo 'Excluded README appeared in staged changes; refusing to commit.' >&2; exit 3 ;;
  esac
done < <(git diff --cached --name-only -z)
if [[ "$PREPARE_ONLY" == 1 ]]; then
  printf '\nPrepared local branch %s; changes staged for review.\n' "$RESEARCH"
  printf 'Preserved original commit locally as %s and annotated tag %s.\n' "$LEGACY" "$TAG"
  printf 'No commit, push, or master promotion performed. README_LEGACY.md excluded.\n'
  exit 0
fi
git commit -m 'feat(eq16): add Gaussian return-sensitivity DR-PETS with modern PETS benchmarks' \
  -m 'Preserve the original density-flow prototype without rewriting history. Add the independent Eq16 benchmark, PETS reference trace, corrected physics, resumable split study and validated AWS deployment.' \
  -m 'Co-Authored-By: Warp <agent@warp.dev>'
if [[ "$LOCAL_ONLY" == 0 ]]; then
  git push origin "$LEGACY"
  git push origin "refs/tags/$TAG"
  git push origin "$RESEARCH"
else
  echo 'Committed locally; no remote refs were pushed.'
fi
printf 'Master was NOT modified. Review and publish separately; never force-push this migration.\n'
