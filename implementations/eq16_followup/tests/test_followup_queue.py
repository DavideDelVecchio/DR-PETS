"""Offline regressions for the authorized single-node bridge/curves sequence."""
import importlib
import time
import unittest

fixtures = importlib.import_module("test_aws_deployment")


class TestBridgeCurvesQueue(unittest.TestCase):
    executable = fixtures.TestAWSDeployment.executable
    run_script = fixtures.TestAWSDeployment.run_script
    calls = fixtures.TestAWSDeployment.calls

    def setUp(self):
        fixtures.TestAWSDeployment.setUp(self)
        self.env.update(
            SHARD="ireland", STAGE="bridge", FAMILY="all", BRIDGE_ALL="0",
            RUN_CURVES_AFTER_BRIDGE="1", FOLLOWUP_ID="offline-followup",
            PYTHON=str(self.python), S3_REGION="eu-central-1",
            S3_RESULTS_URI="s3://test-bucket/study/ireland/results",
            S3_URI="s3://test-bucket/study/followup/new/ireland/results",
            S3_LOGS_URI="s3://test-bucket/study/followup/new/ireland/logs",
        )

    def test_bridge_then_all_curves_share_one_budget(self):
        result = self.run_script("run_bridge_curves_existing_instance.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [c for c in self.calls("MOCK_PYTHON_LOG") if c[:1] == ["followup.py"]]
        self.assertEqual([c[1] for c in commands], ["bridge", "curves"])
        self.assertNotIn("--all-replicas", commands[0])
        self.assertEqual(commands[1][commands[1].index("--family")+1], "all")
        limits = [int(c[2]) for c in self.calls("MOCK_TIMEOUT_LOG")]
        self.assertTrue(0 < limits[1] <= limits[0] <= 3600)
        self.assertTrue((self.project / "logs/followup_bridge_ireland_exit.txt").exists())
        self.assertTrue((self.project / "logs/followup_curves_all_ireland_exit.txt").exists())
        self.assertFalse(any(c[:1] == ["run_shard.py"] for c in self.calls("MOCK_PYTHON_LOG")))

    def test_bridge_failure_does_not_start_curves(self):
        result = self.run_script("run_bridge_curves_existing_instance.sh", MOCK_SHARD_EXIT="124")
        self.assertEqual(result.returncode, 124)
        commands = [c for c in self.calls("MOCK_PYTHON_LOG") if c[:1] == ["followup.py"]]
        self.assertEqual([c[1] for c in commands], ["bridge"])

    def test_external_deadline_is_not_reset_for_second_stage(self):
        result = self.run_script(
            "run_bridge_curves_existing_instance.sh",
            DRPETS_DEADLINE_EPOCH=str(int(time.time()) + 120),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(0 < int(c[2]) <= 120 for c in self.calls("MOCK_TIMEOUT_LOG")))

    def test_queue_launch_preflights_all_four_parent_replicas(self):
        result = self.run_script("launch_followup_aws.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        heads = [c for c in self.calls() if c[:2] == ["s3api", "head-object"]]
        self.assertEqual(len(heads), 20)
        script = (self.capture / "eu-west-1.sh").read_text()
        self.assertIn("export RUN_CURVES_AFTER_BRIDGE='1'", script)
        self.assertIn("run_bridge_curves_existing_instance.sh", script)

    def test_queue_requires_representative_bridge_and_all_families(self):
        for overrides in ({"STAGE": "audit"}, {"BRIDGE_ALL": "1"}, {"FAMILY": "mass"}):
            with self.subTest(overrides=overrides):
                result = self.run_script("launch_followup_aws.sh", **overrides)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
