"""Follow-up AWS integration is stubbed; architecture checks use real CPU models."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fixtures = importlib.import_module("test_aws_deployment")
from cloud.followup_paths import parse_uri, validate_paths


class TestFollowupPaths(unittest.TestCase):
    def test_separate_parent_and_followup_prefixes_are_valid(self):
        validate_paths(
            "s3://test-bucket/study/frankfurt/results",
            "s3://test-bucket/study/followup/new/frankfurt/results",
            "s3://test-bucket/study/followup/new/frankfurt/logs",
        )

    def test_equal_ancestor_and_descendant_outputs_are_rejected(self):
        parent = "s3://test-bucket/study/results"
        for output in (parent, "s3://test-bucket/study", parent + "/extension"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "overlap"):
                validate_paths(parent, output, "s3://test-bucket/other/logs")

    def test_invalid_or_shell_unsafe_paths_are_rejected(self):
        for uri in (
            "https://test-bucket/study", "s3://test-bucket",
            "s3://test-bucket/study/../results", "s3://test-bucket/study//results",
            "s3://test-bucket/study?query=1", "s3://test-bucket/study'quoted",
        ):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                parse_uri(uri)

    def test_results_and_logs_cannot_overlap_or_use_different_buckets(self):
        for logs in ("s3://out-bucket/new/results/logs", "s3://other-bucket/new/logs"):
            with self.subTest(logs=logs), self.assertRaises(ValueError):
                validate_paths("s3://source-bucket/old/results", "s3://out-bucket/new/results", logs)


class TestParentArchitecture(unittest.TestCase):
    def test_default_gaussian_model_has_three_hidden_layers(self):
        from torch import nn
        from eq16_core import GaussianDynamics
        model = GaussianDynamics(18, 6)
        linear = [layer for layer in model.net if isinstance(layer, nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linear],
                         [(24, 128), (128, 128), (128, 128), (128, 36)])
        self.assertEqual(sum(isinstance(layer, nn.SiLU) for layer in model.net), 3)

    def test_saved_five_member_checkpoint_is_verified(self):
        import torch
        from eq16_core import GaussianDynamics
        from cloud.verify_parent_models import verify_architecture
        from run_benchmark import sha256_file
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "ensemble.pt"
            models = []
            for seed in range(5):
                torch.manual_seed(seed)
                models.append(GaussianDynamics(18, 6).state_dict())
            torch.save(models, checkpoint)
            parent = {
                "seed": 101, "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "config": {"ensemble": 5, "width": 128, "depth": 3},
            }
            result = verify_architecture(parent)
            self.assertEqual(result["hidden_layers"], 3)
            self.assertEqual(result["ensemble_members"], 5)
            self.assertEqual(len(set(result["member_sha256"])), 5)
            parent["config"]["depth"] = 2
            with self.assertRaisesRegex(ValueError, "three-hidden-layer"):
                verify_architecture(parent)


class TestFollowupDeployment(unittest.TestCase):
    executable = fixtures.TestAWSDeployment.executable
    run_script = fixtures.TestAWSDeployment.run_script
    calls = fixtures.TestAWSDeployment.calls

    def setUp(self):
        fixtures.TestAWSDeployment.setUp(self)
        self.env.update(FOLLOWUP_ID="offline-followup", STAGE="audit", FAMILY="actuation")

    def launch(self, shard="frankfurt", **overrides):
        result = self.run_script("launch_followup_aws.sh", SHARD=shard, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        region = "eu-west-1" if shard == "ireland" else "eu-central-1"
        return (self.capture / (region + ".sh")).read_text()

    def runtime(self, **overrides):
        settings = dict(
            PYTHON=str(self.python), S3_RESULTS_URI="s3://test-bucket/study/parents/results",
            S3_URI="s3://test-bucket/study/followup/new/results",
            S3_LOGS_URI="s3://test-bucket/study/followup/new/logs", S3_REGION="eu-central-1",
        )
        settings.update(overrides)
        return self.run_script("run_followup_existing_instance.sh", **settings)

    def test_audit_first_launch_preserves_parent_and_separates_code(self):
        script = self.launch("ireland")
        self.assertIn("export STAGE='audit'", script)
        self.assertIn("export AWS_DEFAULT_REGION='eu-west-1'", script)
        self.assertIn("export S3_REGION='eu-central-1'", script)
        self.assertIn("offline-study/ireland/results", script)
        self.assertIn("offline-study/followup/offline-followup/ireland/results", script)
        self.assertIn("run_followup_existing_instance.sh", script)
        self.assertNotIn("run_shard.py", script)
        self.assertNotIn("export AWS_PROFILE", script)
        self.assertIn("MAX_HOURS*3600+300", script)
        subprocess.run(["/bin/bash", "-n"], input=script, text=True, check=True)
        transfers = [c for c in self.calls() if c[:2] == ["s3", "cp"]]
        self.assertTrue(all(c[3].endswith("/followup/offline-followup/project.zip") for c in transfers))
        heads = [c for c in self.calls() if c[:2] == ["s3api", "head-object"]]
        self.assertEqual(len(heads), 20)

    def test_bridge_default_checks_only_the_representative_parent(self):
        self.launch("ireland", STAGE="bridge")
        heads = [c for c in self.calls() if c[:2] == ["s3api", "head-object"]]
        self.assertEqual(len(heads), 5)
        self.assertTrue(all("development_seed_113/" in c[c.index("--key")+1] for c in heads))

    def test_source_bucket_region_is_independent_of_output_and_compute(self):
        script = self.launch(
            "ireland", S3_REGION="eu-central-1",
            IE_S3_RESULTS_URI="s3://parent-bucket/study/ireland/results",
            MOCK_BUCKET_REGION="eu-west-1", MOCK_PARENT_REGION="eu-west-1",
        )
        self.assertIn("export S3_RESULTS_REGION='eu-west-1'", script)
        self.assertIn("export S3_REGION='eu-central-1'", script)
        self.assertIn("export AWS_DEFAULT_REGION='eu-west-1'", script)
        heads = [c for c in self.calls() if c[:2] == ["s3api", "head-object"]]
        self.assertTrue(all(c[c.index("--region")+1] == "eu-west-1" for c in heads))

    def test_invalid_worker_family_and_budget_settings_fail_closed(self):
        for overrides in ({"WORKERS": "0"}, {"FAMILY": "unknown"}, {"MAX_HOURS": "0"}, {"BRIDGE_ALL": "2"}):
            with self.subTest(overrides=overrides):
                result = self.runtime(**overrides)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.calls(), [])

    def test_duplicate_and_missing_parent_block_launch(self):
        for overrides in ({"MOCK_ACTIVE_FOLLOWUP": "1"}, {"MOCK_MISSING_PARENT": "1"}):
            with self.subTest(overrides=overrides):
                result = self.run_script("launch_followup_aws.sh", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))

    def test_invalid_stage_or_parent_output_overlap_blocks_before_aws(self):
        result = self.run_script("launch_followup_aws.sh", STAGE="development")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])
        result = self.run_script(
            "launch_followup_aws.sh",
            S3_RESULTS_URI="s3://test-bucket/old/results",
            S3_URI="s3://test-bucket/old/results",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_runtime_dispatches_followup_only_and_filters_backups(self):
        result = self.runtime()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls("MOCK_PYTHON_LOG")
        self.assertFalse(any(c[:1] == ["run_shard.py"] for c in calls))
        stage = next(c for c in calls if c[:1] == ["followup.py"])
        self.assertEqual(stage[1], "audit")
        requirements = calls.index(["-m", "pip", "install", "-r", "requirements.txt"])
        tests = calls.index(["-m", "pytest", "-q", "tests"])
        physics = calls.index(["cloud/verify_halfcheetah_mass.py"])
        parent = next(i for i, c in enumerate(calls) if c[:1] == ["cloud/verify_parent_models.py"])
        self.assertLess(requirements, tests)
        self.assertLess(tests, physics)
        self.assertLess(physics, parent)
        self.assertLess(parent, calls.index(stage))
        syncs = [c for c in self.calls() if c[:2] == ["s3", "sync"]]
        writes = [c for c in syncs if not c[2].startswith("s3://")]
        self.assertTrue(all(not c[3].startswith("s3://test-bucket/study/parents/") for c in writes))
        for call in syncs:
            if call[2] == "results/" or call[2] == "s3://test-bucket/study/followup/new/results/":
                self.assertIn("followup_*/*", call)
        self.assertEqual(
            (self.project / "logs/followup_audit_frankfurt_exit.txt").read_text(), "exit_code=0\n",
        )

    def test_bridge_all_and_curve_family_flags(self):
        result = self.runtime(STAGE="bridge", BRIDGE_ALL="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(c for c in self.calls("MOCK_PYTHON_LOG") if c[:1] == ["followup.py"])
        self.assertIn("--all-replicas", command)
        # Reset only recorded calls, not the project/source inputs.
        Path(self.env["MOCK_PYTHON_LOG"]).write_text("")
        result = self.runtime(STAGE="curves", FAMILY="friction")
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(c for c in self.calls("MOCK_PYTHON_LOG") if c[:1] == ["followup.py"])
        self.assertEqual(command[command.index("--family")+1], "friction")
        self.assertTrue((self.project / "logs/followup_curves_friction_frankfurt_exit.txt").exists())

    def test_restore_and_test_failures_skip_experiment(self):
        for flag, expected in (("MOCK_FAIL_RESTORE", 19), ("MOCK_FAIL_PYTEST", 9)):
            with self.subTest(flag=flag):
                result = self.runtime(**{flag: "1"})
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertFalse(any(c[:1] == ["followup.py"] for c in self.calls("MOCK_PYTHON_LOG")))

    def test_final_backup_failure_is_nonzero(self):
        result = self.runtime(MOCK_FAIL_BACKUP="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Final follow-up S3 backup failed", result.stderr)

    def test_shared_deadline_and_timeout_preserve_status(self):
        result = self.runtime(DRPETS_DEADLINE_EPOCH=str(int(time.time()) + 120), MOCK_SHARD_EXIT="124")
        self.assertEqual(result.returncode, 124)
        self.assertTrue(0 < int(self.calls("MOCK_TIMEOUT_LOG")[0][2]) <= 120)
        self.assertEqual(
            (self.project / "logs/followup_audit_frankfurt_exit.txt").read_text(), "exit_code=124\n",
        )

    def test_both_region_preflight_prevents_partial_quota_launch(self):
        result = self.run_script("launch_both_followup_aws.sh", MOCK_IRELAND_QUOTA="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))

    def test_read_only_preflight_creates_no_uploads_or_instances(self):
        result = self.run_script("launch_both_followup_aws.sh", PREFLIGHT_ONLY="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))
        self.assertFalse(any(c[:2] == ["s3", "cp"] for c in self.calls()))

    def test_missing_irish_parent_is_caught_before_frankfurt_launch(self):
        result = self.run_script(
            "launch_both_followup_aws.sh", MOCK_MISSING_PARENT_SHARD="ireland",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))
        self.assertFalse(any(c[:2] == ["s3", "cp"] for c in self.calls()))

    def test_dual_followup_launch_and_global_source_rejection(self):
        result = self.run_script("launch_both_followup_aws.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(c[:2] == ["ec2", "run-instances"] for c in self.calls()), 2)
        before = len(self.calls())
        result = self.run_script(
            "launch_both_followup_aws.sh", S3_RESULTS_URI="s3://test-bucket/ambiguous/results",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(self.calls()), before)


if __name__ == "__main__":
    unittest.main()
