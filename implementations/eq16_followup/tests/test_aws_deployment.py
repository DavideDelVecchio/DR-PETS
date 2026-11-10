"""Offline shell integration tests. Every AWS/Python workload call is a stub."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
AWS_STUB = r'''
import json, os, pathlib, sys
a = sys.argv[1:]
with open(os.environ['MOCK_AWS_LOG'], 'a') as f:
    f.write(json.dumps(a) + '\n')
def value(flag, default=''):
    return a[a.index(flag)+1] if flag in a else default
region = value('--region')
op = tuple(a[:2])
if op == ('s3api', 'get-bucket-location'):
    print(os.environ.get('MOCK_BUCKET_REGION', 'eu-central-1'))
elif op == ('s3api', 'head-object'):
    if os.environ.get('MOCK_MISSING_PARENT'):
        sys.exit(44)
    missing_shard = os.environ.get('MOCK_MISSING_PARENT_SHARD')
    if missing_shard and ('/' + missing_shard + '/results/') in value('--key'):
        sys.exit(44)
    print('839335')
elif op == ('sts', 'get-caller-identity'):
    print('{}')
elif op == ('ec2', 'describe-instance-types'):
    print('8')
elif op == ('ec2', 'describe-instances'):
    print(os.environ.get('MOCK_ACTIVE_FOLLOWUP', '0'))
elif op == ('ec2', 'describe-instance-type-offerings'):
    print(os.environ.get('MOCK_OFFERED', '1'))
elif op == ('service-quotas', 'get-service-quota'):
    print(os.environ.get('MOCK_IRELAND_QUOTA', '8') if region == 'eu-west-1' else '8')
elif op == ('ssm', 'get-parameter'):
    print('ami-' + region)
elif op == ('ec2', 'describe-images'):
    print('/dev/sda1')
elif op == ('ec2', 'run-instances'):
    if region == os.environ.get('MOCK_FAIL_LAUNCH_REGION'):
        print('InsufficientInstanceCapacity', file=sys.stderr)
        sys.exit(17)
    script = pathlib.Path(value('--user-data').removeprefix('file://')).read_text()
    pathlib.Path(os.environ['MOCK_CAPTURE'], region + '.sh').write_text(script)
    print('i-mock-' + region)
elif a[:1] == ['s3']:
    expected_region = os.environ.get('MOCK_EXPECT_S3_REGION', 'eu-central-1')
    if a[2].startswith('s3://parent-bucket/'):
        expected_region = os.environ.get('MOCK_PARENT_REGION', 'eu-west-1')
    if region != expected_region:
        print('Wrong S3 region: ' + region, file=sys.stderr)
        sys.exit(88)
    if a[1] == 'cp' and a[2].startswith('s3://') and os.environ.get('MOCK_FAIL_DOWNLOAD'):
        sys.exit(18)
    if a[1] == 'sync' and a[2].startswith('s3://') and os.environ.get('MOCK_FAIL_RESTORE'):
        sys.exit(19)
    if a[1] == 'sync' and not a[2].startswith('s3://') and os.environ.get('MOCK_FAIL_BACKUP'):
        sys.exit(20)
else:
    print('Unexpected AWS operation: ' + repr(a), file=sys.stderr)
    sys.exit(99)
'''
PYTHON_STUB = r'''
import json, os, pathlib, sys
a = sys.argv[1:]
with open(os.environ['MOCK_PYTHON_LOG'], 'a') as f:
    f.write(json.dumps(a) + '\n')
if a and a[0].endswith('followup_paths.py'):
    os.execv(sys.executable, [sys.executable] + a)
if a[:2] == ['-m', 'pytest'] and os.environ.get('MOCK_FAIL_PYTEST'):
    sys.exit(9)
if a[:1] == ['run_shard.py']:
    pathlib.Path('results/mock-completed.txt').write_text('complete')
    print('Mock shard completed')
    sys.exit(int(os.environ.get('MOCK_SHARD_EXIT', '0')))
if a[:1] == ['followup.py']:
    pathlib.Path('results/followup_mock').mkdir(exist_ok=True)
    pathlib.Path('results/followup_mock/complete.json').write_text('{}')
    print('Mock follow-up completed')
    sys.exit(int(os.environ.get('MOCK_SHARD_EXIT', '0')))
'''
TIMEOUT_STUB = r'''
import json, os, sys
with open(os.environ['MOCK_TIMEOUT_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\n')
os.execv(sys.argv[4], sys.argv[4:])
'''


class TestAWSDeployment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="drpets-offline-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.project = self.base / "renamed_study"
        shutil.copytree(ROOT / "cloud", self.project / "cloud")
        shutil.copytree(ROOT / "legacy", self.project / "legacy")
        (self.project / "requirements.txt").write_text("mock-dependency\n")
        self.bin = self.base / "bin"
        self.bin.mkdir()
        self.capture = self.base / "capture"
        self.capture.mkdir()
        self.env = {
            "PATH": str(self.bin) + os.pathsep + os.defpath,
            "HOME": str(self.base),
            "AWS_PROFILE": "default",
            "AWS_EC2_METADATA_DISABLED": "true",
            "S3_BUCKET": "test-bucket",
            "INSTANCE_PROFILE": "test-profile",
            "RUN_ID": "offline-study",
            "SHARD": "frankfurt",
            "MAX_HOURS": "1",
            "SYNC_INTERVAL": "1",
            "LOCAL_PYTHON": sys.executable,
            "MOCK_AWS_LOG": str(self.base / "aws.jsonl"),
            "MOCK_PYTHON_LOG": str(self.base / "python.jsonl"),
            "MOCK_TIMEOUT_LOG": str(self.base / "timeout.jsonl"),
            "MOCK_CAPTURE": str(self.capture),
        }
        self.executable("aws", AWS_STUB)
        self.python = self.executable("mock-python", PYTHON_STUB)
        self.executable("timeout", TIMEOUT_STUB)
        self.executable("zip", "import pathlib,sys\npathlib.Path(sys.argv[2]).touch()\n")
        self.executable("unzip", "pass\n")

    def executable(self, name, code):
        path = self.bin / name
        path.write_text("#!" + sys.executable + "\n" + code)
        path.chmod(0o700)
        return path

    def run_script(self, name, **overrides):
        return subprocess.run(
            ["/bin/bash", str(self.project / "cloud" / name)],
            env={**self.env, **overrides}, capture_output=True, text=True, timeout=20,
        )

    def calls(self, name="MOCK_AWS_LOG"):
        path = Path(self.env[name])
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def launch(self, shard="frankfurt", **overrides):
        result = self.run_script("launch_shard_aws.sh", SHARD=shard, **overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        region = "eu-west-1" if shard == "ireland" else "eu-central-1"
        return (self.capture / (region + ".sh")).read_text()

    def bootstrap_body(self, script, **overrides):
        # Execute only the real fail-fast subshell, never watchdog/shutdown.
        body = script.split("set +e\n(\n", 1)[1].split("\n) >", 1)[0]
        sandbox = self.base / "sandbox"
        project = sandbox / self.project.name
        (project / "cloud").mkdir(parents=True, exist_ok=True)
        marker = self.base / "shard-ran"
        (project / "cloud/run_on_existing_instance.sh").write_text(
            '#!/bin/bash\nprintf complete >"$MOCK_RAN"\n'
        )
        body = body.replace("/opt/drpets", str(sandbox))
        body = body.replace("/tmp/drpets-project.zip", str(self.base / "download.zip"))
        sandbox.mkdir(exist_ok=True)
        result = subprocess.run(
            ["/bin/bash", "-c", body],
            env={
                **self.env, "S3_REGION": "eu-central-1", "PROJECT_KEY": "project.zip",
                "PREFIX": "drpets-eight/offline-study/ireland", "MOCK_RAN": str(marker),
                **overrides,
            }, capture_output=True, text=True, timeout=10,
        )
        return result, marker

    def test_both_regions_keep_s3_and_compute_regions_separate(self):
        for shard, region in [("frankfurt", "eu-central-1"), ("ireland", "eu-west-1")]:
            with self.subTest(shard=shard):
                script = self.launch(shard)
                self.assertIn("export AWS_DEFAULT_REGION='" + region + "'", script)
                self.assertIn("export S3_REGION='eu-central-1'", script)
                self.assertIn("cd 'renamed_study'", script)
                self.assertIn("MAX_HOURS*3600+300", script)
                self.assertIn("export DRPETS_DEADLINE_EPOCH=", script)
                self.assertIn("set +e\n(\n  set -euo pipefail", script)
                subprocess.run(["/bin/bash", "-n"], input=script, text=True, check=True)
        for call in self.calls():
            if call[:1] == ["s3"]:
                self.assertEqual(call[call.index("--region") + 1], "eu-central-1")
            if call[:2] == ["ec2", "run-instances"]:
                self.assertNotIn("--instance-market-options", call)

    def test_regional_overrides_and_spot_arguments(self):
        self.launch(
            "ireland", MODE="spot", IE_SUBNET_ID="subnet-ireland",
            IE_SECURITY_GROUP_ID="sg-ireland", IE_AMI_ID="ami-override",
            FR_SUBNET_ID="subnet-frankfurt",
        )
        call = next(c for c in self.calls() if c[:2] == ["ec2", "run-instances"])
        for flag, expected in [
            ("--subnet-id", "subnet-ireland"), ("--security-group-ids", "sg-ireland"),
            ("--image-id", "ami-override"),
        ]:
            self.assertEqual(call[call.index(flag) + 1], expected)
        self.assertIn("--instance-market-options", call)
        self.assertFalse(any(c[:2] == ["ssm", "get-parameter"] for c in self.calls()))

    def test_dual_launch_preflights_before_creating_either_instance(self):
        result = self.run_script("launch_both_aws.sh", MOCK_IRELAND_QUOTA="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))

    def test_dual_launch_succeeds_in_both_regions(self):
        result = self.run_script("launch_both_aws.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        launch_indices = [i for i, c in enumerate(calls) if c[:2] == ["ec2", "run-instances"]]
        self.assertEqual(len(launch_indices), 2)
        quota_indices = [i for i, c in enumerate(calls) if c[:2] == ["service-quotas", "get-service-quota"]]
        self.assertTrue(max(quota_indices) < min(launch_indices))

    def test_dual_launch_rejects_global_region_specific_ids(self):
        for name in ("SUBNET_ID", "SECURITY_GROUP_ID", "AMI_ID"):
            with self.subTest(name=name):
                result = self.run_script("launch_both_aws.sh", **{name: "regional-id"})
                self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_second_region_capacity_failure_preserves_first_instance_details(self):
        result = self.run_script("launch_both_aws.sh", MOCK_FAIL_LAUNCH_REGION="eu-west-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("i-mock-eu-central-1", result.stdout)
        self.assertTrue((self.capture / "eu-central-1.sh").exists())
        self.assertFalse((self.capture / "eu-west-1.sh").exists())
        self.assertFalse(any(c[:2] == ["ec2", "terminate-instances"] for c in self.calls()))

    def test_missing_offering_stops_before_launch(self):
        result = self.run_script("launch_both_aws.sh", MOCK_OFFERED="0")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c[:2] == ["ec2", "run-instances"] for c in self.calls()))

    def test_preflight_uses_spot_quota_when_requested(self):
        result = self.run_script("check_dual_region.sh", MODE="spot")
        self.assertEqual(result.returncode, 0, result.stderr)
        quotas = [c for c in self.calls() if c[:2] == ["service-quotas", "get-service-quota"]]
        self.assertEqual(len(quotas), 2)
        self.assertTrue(all(c[c.index("--quota-code")+1] == "L-3819A6DF" for c in quotas))

    def test_failed_download_and_restore_do_not_start_the_shard(self):
        script = self.launch("ireland")
        for flag, expected in [("MOCK_FAIL_DOWNLOAD", 18), ("MOCK_FAIL_RESTORE", 19)]:
            with self.subTest(flag=flag):
                result, marker = self.bootstrap_body(script, **{flag: "1"})
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertFalse(marker.exists())

    def test_fresh_or_successful_restore_starts_the_shard(self):
        result, marker = self.bootstrap_body(self.launch("ireland"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(marker.is_file())

    def test_runtime_uses_full_setup_then_physics_then_workload(self):
        result = self.run_script(
            "run_on_existing_instance.sh", PYTHON=str(self.python),
            S3_URI="s3://test-bucket/drpets-eight/offline-study/frankfurt",
            S3_REGION="eu-central-1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls("MOCK_PYTHON_LOG")
        requirements = calls.index(["-m", "pip", "install", "-r", "requirements.txt"])
        tests = calls.index(["-m", "pytest", "-q", "tests"])
        physics = calls.index(["cloud/verify_halfcheetah_mass.py"])
        workload = next(i for i, c in enumerate(calls) if c[:1] == ["run_shard.py"])
        self.assertLess(requirements, tests)
        self.assertLess(tests, physics)
        self.assertLess(physics, workload)
        self.assertTrue((self.project / "results/mock-completed.txt").is_file())
        self.assertEqual((self.project / "logs/frankfurt_development_exit.txt").read_text(), "exit_code=0\n")
        transfers = [c for c in self.calls() if c[:2] == ["s3", "sync"]]
        self.assertTrue(any(c[2] == "results/" for c in transfers))
        self.assertTrue(any(c[2] == "logs/" for c in transfers))
        seconds = int(self.calls("MOCK_TIMEOUT_LOG")[0][2])
        self.assertTrue(0 < seconds <= 3600)

    def test_runtime_setup_failure_preserves_status_and_skips_workload(self):
        result = self.run_script("run_on_existing_instance.sh", PYTHON=str(self.python), MOCK_FAIL_PYTEST="1")
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertFalse(any(c[:1] == ["run_shard.py"] for c in self.calls("MOCK_PYTHON_LOG")))
        self.assertEqual((self.project / "logs/frankfurt_development_exit.txt").read_text(), "exit_code=9\n")

    def test_runtime_final_backup_failure_is_not_success(self):
        result = self.run_script(
            "run_on_existing_instance.sh", PYTHON=str(self.python),
            S3_URI="s3://test-bucket/drpets-eight/offline-study/frankfurt",
            S3_REGION="eu-central-1", MOCK_FAIL_BACKUP="1",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Final S3 backup failed", result.stderr)

    def test_runtime_timeout_status_is_preserved(self):
        result = self.run_script("run_on_existing_instance.sh", PYTHON=str(self.python), MOCK_SHARD_EXIT="124")
        self.assertEqual(result.returncode, 124)
        self.assertEqual((self.project / "logs/frankfurt_development_exit.txt").read_text(), "exit_code=124\n")

    def test_shared_deadline_limits_remaining_runtime(self):
        result = self.run_script(
            "run_on_existing_instance.sh", PYTHON=str(self.python),
            DRPETS_DEADLINE_EPOCH=str(int(time.time()) + 120),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(0 < int(self.calls("MOCK_TIMEOUT_LOG")[0][2]) <= 120)

    def test_expired_shared_deadline_skips_workload(self):
        result = self.run_script(
            "run_on_existing_instance.sh", PYTHON=str(self.python),
            DRPETS_DEADLINE_EPOCH=str(int(time.time()) - 1),
        )
        self.assertEqual(result.returncode, 124)
        self.assertFalse(any(c[:1] == ["run_shard.py"] for c in self.calls("MOCK_PYTHON_LOG")))

    def test_invalid_stage_and_missing_interpreter_fail_before_work(self):
        result = self.run_script("launch_shard_aws.sh", STAGE="invalid")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])
        result = self.run_script("run_on_existing_instance.sh", PYTHON=str(self.base / "missing"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("DLAMI PyTorch interpreter missing", result.stderr)

    def test_bucket_location_normalization_and_explicit_override(self):
        helper = self.project / "cloud/aws_common.sh"
        script = 'source "$1"; unset AWS_PROFILE; resolve_s3_region; printf "%s" "$S3_REGION"'
        for location, expected in [("None", "us-east-1"), ("EU", "eu-west-1")]:
            with self.subTest(location=location):
                result = subprocess.run(
                    ["/bin/bash", "-euc", script, "test", str(helper)],
                    env={**self.env, "MOCK_BUCKET_REGION": location}, capture_output=True, text=True, check=True,
                )
                self.assertEqual(result.stdout, expected)
        before = len(self.calls())
        result = subprocess.run(
            ["/bin/bash", "-euc", script, "test", str(helper)],
            env={**self.env, "S3_REGION": "eu-central-1"}, capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout, "eu-central-1")
        self.assertEqual(before, len(self.calls()))

    def test_download_uses_bucket_region_and_selected_local_interpreter(self):
        result = self.run_script("download_and_analyze.sh", PYTHON=str(self.python))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len([c for c in calls if c[:2] == ["s3", "sync"]]), 2)
        self.assertTrue(self.calls("MOCK_PYTHON_LOG")[0][0].endswith("/analyze_shards.py"))
        analysis_call = self.calls("MOCK_PYTHON_LOG")[0]
        self.assertEqual(
            analysis_call[analysis_call.index("--root") + 1],
            str(self.project / "results/downloads/offline-study"),
        )

    def test_legacy_generated_startup_syntax(self):
        env = {
            **self.env, "AWS_REGION": "eu-west-1", "S3_REGION": "eu-central-1",
            "S3_PREFIX": "test-run", "RUN_ARGS": "--task halfcheetah",
            "RUN_ARGS_B64": "dGVzdA==", "PROJECT_NAME": self.project.name,
            "GET_URL": "https://example.invalid/input", "PUT_URL": "https://example.invalid/output",
        }
        for name in ("launch_aws.sh", "launch_aws_no_instance_role.sh"):
            with self.subTest(launcher=name):
                text = (self.project / "legacy" / name).read_text()
                template = text.split('cat > "$TMP/userdata.sh" <<EOF\n', 1)[1].split("\nEOF\n", 1)[0]
                rendered = subprocess.run(
                    ["/bin/bash", "-uc", "cat <<EOF\n" + template + "\nEOF\n"],
                    env=env, capture_output=True, text=True, check=True,
                ).stdout
                inner = rendered.split("cat > /opt/drpets/entry.sh <<'INNER'\n", 1)[1].split("\nINNER\n", 1)[0]
                for script in (rendered, inner):
                    subprocess.run(["/bin/bash", "-n"], input=script, text=True, check=True)

    def test_legacy_scripts_keep_dlami_and_bash_fixes(self):
        for name in ("launch_aws.sh", "launch_aws_no_instance_role.sh"):
            text = (self.project / "legacy" / name).read_text()
            self.assertIn("PYTHON=/opt/pytorch/bin/python", text)
            self.assertNotIn("/opt/conda/envs", text)
            self.assertIn('pip install -r requirements.txt', text)
            self.assertIn('cloud/verify_halfcheetah_mass.py', text)
            self.assertIn('${OPTIONS[@]+"${OPTIONS[@]}"}', text)
            subprocess.run(["/bin/bash", "-n", str(self.project / "legacy" / name)], check=True)

    def test_deployment_defaults_work_without_manual_exports(self):
        (self.project / "cloud/deployment.env").write_text(
            'export AWS_PROFILE="${AWS_PROFILE:-default}"\n'
            'export RUN_ID="${RUN_ID:-offline-study}"\n'
            'export S3_BUCKET="${S3_BUCKET:-test-bucket}"\n'
            'export INSTANCE_PROFILE="${INSTANCE_PROFILE:-offline-profile}"\n'
        )
        for key in ("AWS_PROFILE", "RUN_ID", "S3_BUCKET", "INSTANCE_PROFILE"):
            self.env.pop(key)
        self.launch("ireland")
        call = next(c for c in self.calls() if c[:2] == ["ec2", "run-instances"])
        self.assertEqual(
            call[call.index("--iam-instance-profile") + 1],
            "Name=offline-profile",
        )
        self.assertIn("offline-study", call[call.index("--tag-specifications") + 1])
        transfers = [c for c in self.calls() if c[:2] == ["s3", "cp"]]
        self.assertTrue(any("s3://test-bucket/drpets-eight/offline-study/project.zip" in c for c in transfers))

    def test_deployment_config_preserves_explicit_environment_overrides(self):
        self.launch(
            "ireland", RUN_ID="another-study", INSTANCE_PROFILE="another-profile",
            IE_INSTANCE_TYPE="g6.2xlarge",
        )
        call = next(c for c in self.calls() if c[:2] == ["ec2", "run-instances"])
        self.assertEqual(call[call.index("--iam-instance-profile") + 1], "Name=another-profile")
        self.assertEqual(call[call.index("--instance-type") + 1], "g6.2xlarge")
        self.assertIn("another-study", call[call.index("--tag-specifications") + 1])

    def test_missing_explicit_deployment_config_fails_before_aws(self):
        result = self.run_script(
            "launch_shard_aws.sh", DEPLOYMENT_CONFIG=str(self.base / "missing.env"),
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_instance_runtime_does_not_load_local_profile_defaults(self):
        text = (self.project / "cloud/run_on_existing_instance.sh").read_text()
        self.assertNotIn("load_deployment_config", text)
        self.assertNotIn("export AWS_PROFILE", self.launch("ireland"))

    def test_example_policy_is_scoped_to_one_configured_study(self):
        policy = json.loads((ROOT / "cloud/iam_policy.example.json").read_text())
        statements = {item["Sid"]: item for item in policy["Statement"]}
        self.assertEqual(
            statements["ReadWriteOnlyThisResearchRun"]["Resource"],
            "arn:aws:s3:::REPLACE_BUCKET/drpets-eight/REPLACE_RUN_ID/*",
        )
        self.assertEqual(
            set(statements["ReadWriteOnlyThisResearchRun"]["Action"]),
            {"s3:GetObject", "s3:PutObject"},
        )
        self.assertEqual(
            statements["ListOnlyThisResearchRun"]["Condition"]["StringLike"]["s3:prefix"],
            ["drpets-eight/REPLACE_RUN_ID/*"],
        )
        self.assertEqual(
            {a for item in policy["Statement"] for a in item["Action"]},
            {"s3:GetBucketLocation", "s3:GetObject", "s3:PutObject", "s3:ListBucket"},
        )


if __name__ == "__main__":
    unittest.main()
