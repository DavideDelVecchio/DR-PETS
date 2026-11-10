# History-preserving migration

The old `master` head at the time this package was prepared was:

`a4920349d32aec8a4c7f360049979c4848a31239` (2025-05-26).

Recommended archive branch: `legacy/state-action-density-2025`.
Recommended **annotated** tag on the exact original commit: `legacy-density-score-v0.1.0`.
No archival README commit is created: the historical branch preserves the original tree exactly.
Suggested modern commit: `feat(eq16): introduce probabilistic return-sensitivity DR-PETS and multi-task PETS benchmarks`.

The migration helper at `scripts/prepare_migration.sh` defaults to `LOCAL_ONLY=1`, `PREPARE_ONLY=1`, and `EXCLUDE_LEGACY_README=1`. It preserves the exact original tree on a local legacy branch and annotated tag and stages a review branch `research/eq16-benchmarks`. It does not rename/import the excluded legacy README, commit, push or rewrite master in this mode.

Invoke the script from the supplied migration package, or set `MIGRATION_BUNDLE` to the original package containing `new_master/` when using the installed copy. The script verifies the expected origin and audited master SHA, a clean checkout, and absence of conflicting branch/tag refs. Optional expected-ref overrides are for deliberately reviewed changes or disposable tests, not automatic bypasses.

After tests and staged-file review, make the local commit with `Co-Authored-By: Warp <agent@warp.dev>`. Preserve `README_LEGACY.md` exclusion in both the staged paths and resulting tree. Publication requires a separate explicit action and valid GitHub credentials. Only after publication/review can `scripts/promote_master.sh` fast-forward master (with its own explicit confirmation and ref checks); it is not part of preparation.

Important: GitHub Actions or branch protection may require merging a PR instead; you can use `gh pr create --base master --head research/eq16-benchmarks` followed by normal review and merge. Avoid automatic release labels until the HalfCheetah CUDA benchmark is validated. The historical tag is about historical provenance, **not** an upstream paper version or validated equivalence.

Other branches `main`, `full-implementation`, `full-metal-device`, `surrogate-rewards` are intentionally left untouched.
