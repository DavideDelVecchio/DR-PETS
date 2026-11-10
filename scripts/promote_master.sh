#!/usr/bin/env bash
# Explicit opt-in fast-forward promotion of tested research/eq16-benchmarks to master.
# Does not force push. Existing historical master preserved on legacy branch + tag.
set -euo pipefail
REPO="${1:-$PWD}"
[[ "${PROMOTE_MASTER:-}" == YES ]] || {
 echo 'Set PROMOTE_MASTER=YES to explicitly promote the reviewed research branch to master.' >&2; exit 2;
}
cd "$REPO"
[[ -d .git ]] || { echo 'Not a local git clone.' >&2; exit 2; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Dirty working tree.' >&2; exit 2; }
git fetch origin master research/eq16-benchmarks --tags
BASE="$(git rev-parse refs/remotes/origin/master)"
TARGET="$(git rev-parse refs/remotes/origin/research/eq16-benchmarks)"
[[ "$BASE" == 'a4920349d32aec8a4c7f360049979c4848a31239' ]] || {
 echo 'Master has moved; inspect before promoting.' >&2; exit 3;
}
git show-ref --verify --quiet refs/remotes/origin/legacy/state-action-density-2025 || {
 echo 'Legacy preservation branch missing.' >&2; exit 3;
}
git ls-remote --exit-code --tags origin legacy-density-score-v0.1.0 >/dev/null || {
 echo 'Original historical tag missing on origin.' >&2; exit 3;
}
git merge-base --is-ancestor "$BASE" "$TARGET" || {
 echo 'Research branch is not a fast-forward descendant of master.' >&2; exit 3;
}
# No --force. Recommended alternative: merge the reviewed PR via GitHub's UI.
git push origin "$TARGET:refs/heads/master"
echo "Promoted $TARGET to master (fast forward). Historical tag and legacy branch remain."
