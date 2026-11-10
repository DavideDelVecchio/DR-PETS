"""Migration regressions use disposable local Git repositories, never GitHub."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_migration.sh"
LEGACY = "legacy/state-action-density-2025"
RESEARCH = "research/eq16-benchmarks"
TAG = "legacy-density-score-v0.1.0"


def git(directory, *args):
    return subprocess.run(
        ["git", "--no-pager", "-C", str(directory), *args],
        check=True, capture_output=True, text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


class TestMigration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drpets-migration-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seed = self.root / "seed"
        seed.mkdir()
        git(seed, "init")
        git(seed, "symbolic-ref", "HEAD", "refs/heads/master")
        self.identity(seed)
        (seed / "README.md").write_text("Original historical README\n")
        (seed / "old.py").write_text("historical = True\n")
        git(seed, "add", ".")
        git(seed, "commit", "-m", "Original fixture")
        self.base = git(seed, "rev-parse", "HEAD")
        self.remote = self.root / "origin.git"
        subprocess.run(
            ["git", "--no-pager", "clone", "--bare", str(seed), str(self.remote)],
            check=True, capture_output=True,
        )
        self.repo = self.root / "checkout"
        subprocess.run(
            ["git", "--no-pager", "clone", str(self.remote), str(self.repo)],
            check=True, capture_output=True,
        )
        self.identity(self.repo)
        self.bundle = self.root / "bundle"
        self.payload = self.bundle / "new_master"
        self.payload.mkdir(parents=True)
        (self.bundle / "README_LEGACY.md").write_text("EXCLUDED_ARCHIVAL_TEXT\n")
        (self.payload / "README.md").write_text("Modern benchmark\n")
        (self.payload / "new.py").write_text("modern = True\n")
        (self.payload / ".gitignore").write_text("README_LEGACY.md\nresults/\nlogs/\n*.pt\n")
        (self.payload / "README_LEGACY.md").write_text("EXCLUDED_ARCHIVAL_TEXT\n")
        (self.payload / "nested").mkdir()
        (self.payload / "nested/README_LEGACY.md").write_text("EXCLUDED_ARCHIVAL_TEXT\n")
        for name in ("results/output.csv", "logs/private.log", "cloud/deployment.env", "cloud/followup.env", "cloud/iam_policy.local.json", "baseline_eight_replica_figures/plot.svg"):
            path = self.payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PRIVATE_FIXTURE\n")
        self.env = {
            **os.environ, "MIGRATION_BUNDLE": str(self.bundle),
            "EXPECTED_REMOTE": str(self.remote), "EXPECTED_MASTER_SHA": self.base,
            "LOCAL_ONLY": "1", "PREPARE_ONLY": "1", "EXCLUDE_LEGACY_README": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.before_remote = git(self.remote, "show-ref")

    def identity(self, directory):
        git(directory, "config", "user.name", "Migration Test")
        git(directory, "config", "user.email", "migration@example.invalid")
        git(directory, "config", "commit.gpgsign", "false")
        git(directory, "config", "tag.gpgsign", "false")

    def prepare(self, **overrides):
        return subprocess.run(
            ["/bin/bash", str(SCRIPT), str(self.repo)],
            env={**self.env, **overrides},
            capture_output=True, text=True, timeout=30,
        )

    def assert_remote_unchanged(self):
        self.assertEqual(git(self.remote, "show-ref"), self.before_remote)

    def test_preparation_preserves_history_without_commits_or_pushes(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(self.repo, "symbolic-ref", "--short", "HEAD"), RESEARCH)
        self.assertEqual(git(self.repo, "rev-parse", "HEAD"), self.base)
        self.assertEqual(git(self.repo, "rev-parse", "master"), self.base)
        self.assertEqual(git(self.repo, "rev-parse", LEGACY), self.base)
        self.assertEqual(git(self.repo, "rev-parse", TAG + "^{commit}"), self.base)
        self.assertEqual(git(self.repo, "cat-file", "-t", TAG), "tag")
        self.assertEqual(git(self.repo, "show", LEGACY + ":README.md"), "Original historical README")
        self.assertIn("new.py", git(self.repo, "diff", "--cached", "--name-only"))
        self.assert_remote_unchanged()

    def test_excludes_legacy_readme_private_files_and_generated_artifacts(self):
        result = self.prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        names = git(self.repo, "ls-files").splitlines()
        self.assertFalse(any(Path(name).name == "README_LEGACY.md" for name in names))
        self.assertFalse(any(name.startswith(("results/", "logs/")) for name in names))
        self.assertNotIn("cloud/deployment.env", names)
        self.assertNotIn("cloud/iam_policy.local.json", names)
        self.assertNotIn("cloud/followup.env", names)
        self.assertFalse(any(name.startswith("baseline_eight_replica_figures/") for name in names))
        for name in names:
            self.assertNotIn("EXCLUDED_ARCHIVAL_TEXT", (self.repo / name).read_text())
        self.assertEqual((self.bundle / "README_LEGACY.md").read_text(), "EXCLUDED_ARCHIVAL_TEXT\n")
        self.assert_remote_unchanged()

    def test_dirty_checkout_is_rejected(self):
        (self.repo / "uncommitted.txt").write_text("keep me\n")
        result = self.prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not clean", result.stderr)
        self.assertEqual(git(self.repo, "symbolic-ref", "--short", "HEAD"), "master")
        self.assert_remote_unchanged()

    def test_changed_audited_master_is_rejected(self):
        result = self.prepare(EXPECTED_MASTER_SHA="0" * 40)
        self.assertEqual(result.returncode, 3)
        self.assertIn("REFUSING automatic migration", result.stderr)
        self.assertEqual(git(self.repo, "symbolic-ref", "--short", "HEAD"), "master")
        self.assert_remote_unchanged()

    def test_wrong_remote_is_rejected(self):
        result = self.prepare(EXPECTED_REMOTE="https://example.invalid/another/repo.git")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unexpected remote", result.stderr)
        self.assert_remote_unchanged()

    def test_legacy_branch_collision_is_rejected(self):
        git(self.repo, "branch", LEGACY, self.base)
        result = self.prepare()
        self.assertEqual(result.returncode, 3)
        self.assertIn("Legacy branch already exists", result.stderr)
        self.assert_remote_unchanged()

    def test_research_branch_collision_is_rejected(self):
        git(self.repo, "branch", RESEARCH, self.base)
        result = self.prepare()
        self.assertEqual(result.returncode, 3)
        self.assertIn("Research branch already exists", result.stderr)
        self.assert_remote_unchanged()

    def test_tag_collision_is_rejected(self):
        git(self.repo, "tag", "-a", TAG, "-m", "existing tag")
        result = self.prepare()
        self.assertEqual(result.returncode, 3)
        self.assertIn("Historical tag already exists", result.stderr)
        self.assert_remote_unchanged()

    def test_local_commit_mode_retains_attribution_and_never_pushes(self):
        result = self.prepare(PREPARE_ONLY="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(git(self.repo, "rev-parse", "HEAD"), self.base)
        self.assertIn("Co-Authored-By: Warp <agent@warp.dev>", git(self.repo, "log", "-1", "--format=%B"))
        self.assertEqual(git(self.repo, "status", "--porcelain"), "")
        self.assert_remote_unchanged()

    def test_publish_cannot_happen_in_prepare_only_mode(self):
        result = self.prepare(LOCAL_ONLY="0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Publishing requires", result.stderr)
        self.assert_remote_unchanged()

    def test_legacy_readme_exclusion_cannot_be_disabled(self):
        result = self.prepare(EXCLUDE_LEGACY_README="0")
        self.assertEqual(result.returncode, 2)
        self.assert_remote_unchanged()

    def test_promotion_requires_explicit_opt_in(self):
        env = dict(self.env)
        env.pop("PROMOTE_MASTER", None)
        result = subprocess.run(
            ["/bin/bash", str(ROOT / "scripts/promote_master.sh"), str(self.repo)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assert_remote_unchanged()


if __name__ == "__main__":
    unittest.main()
