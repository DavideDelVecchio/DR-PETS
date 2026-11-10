# DR-PETS Eq. (16): ready-to-run Frankfurt + Ireland split

This is an independent Python/PyTorch implementation. **It does not use code from the user's GitHub DR-PETS repository.** It derives from the standalone Eq. (16) implementation in the previous benchmark package.

**Status:** this repository layout includes scientific CPU tests and offline AWS integration tests. Deployment fixes were ported from the verified standalone workflow; this migration does not itself provision or update any AWS instances. Validate the complete test suite and a separately authorized CUDA release check before a new study.

## Repository layout and deployment compatibility

- Run local commands from `implementations/eq16` in the repository. Automated launchers archive this implementation directory only and extract it as `/opt/drpets/eq16`; the rest of the repository is not required on the instance.
- EC2 uses `/opt/pytorch/bin/python` by default, validates pip/PyTorch, installs all of `requirements.txt`, runs tests and the live `cloud/verify_halfcheetah_mass.py` check, then starts the assigned shard. No silent system-Python fallback is used.
- `S3_REGION` is the bucket location, independently of Frankfurt/Ireland compute placement. Set it explicitly or allow `s3:GetBucketLocation` discovery. Cross-region transfer charges may apply.
- Configure `FR_SUBNET_ID` / `IE_SUBNET_ID`, `FR_SECURITY_GROUP_ID` / `IE_SECURITY_GROUP_ID`, and `FR_AMI_ID` / `IE_AMI_ID` separately. The dual launcher rejects ambiguous global network/AMI overrides.
- Source/resume download failures stop bootstrap. Both results and stage-specific logs are backed up periodically; final backup failures are not reported as success. A shared deadline counts bootstrap/setup time and allows five minutes for final cleanup before the independent shutdown watchdog.
- The public repository includes no account-specific instance profile, bucket, policy or active-study defaults. Export your settings, or copy `cloud/deployment.env.example` to ignored `cloud/deployment.env` and configure a trusted local file. Explicit environment variables take precedence; EC2 does not load the laptop's AWS profile configuration.
- Use a new run ID/prefix with its own scoped S3 policy for changed scientific code. Do not overwrite source/checkpoints or resume CSVs from an already running standalone study merely because this repository has a new layout.

For Intel-macOS CPU testing only, `constraints-macos-intel-test.txt` records compatible wheel versions. Do not apply those pins to a Linux/CUDA DLAMI. Generated results, credentials, local configuration and `README_LEGACY.md` are excluded from source control.

## 1. Workload allocation

| Region/shard | Independent training seeds | Optional 1,000-step bridge |
| --- | --- | --- |
| Frankfurt (`eu-central-1`) | 101, 103, 107, 109 | 101 |
| Ireland (`eu-west-1`) | 113, 127, 131, 137 | 113 |

See `STUDY_PLAN.md` for parameter settings, exact counts, statistical interpretation and cost assumptions.

## 2. Easiest path: use GPU EC2 machines you already have

Copy the entire project directory to **each machine** using your existing transfer method (SCP/SSM/SSH). Each machine runs only its assigned seeds:

Frankfurt:

```bash
cd implementations/eq16
SHARD=frankfurt STAGE=development MAX_HOURS=24 \
  bash cloud/run_on_existing_instance.sh
```

Ireland:

```bash
cd implementations/eq16
SHARD=ireland STAGE=development MAX_HOURS=24 \
  bash cloud/run_on_existing_instance.sh
```

If you have configured S3 credentials/a scoped EC2 role on those nodes, you can set `S3_URI=s3://YOUR_BUCKET/drpets-eight/YOUR_RUN_ID/frankfurt` (or `/ireland`) before invoking to sync results every two minutes. Otherwise `results/` and `logs/` remain **on the instance and must be copied off before termination**. Running on an *already existing* instance needs no new security group or instance profile, but your access method and results persistence are your responsibility.

**Important:** direct-on-instance script does not automatically terminate an already-existing EC2 instance after it exits. Check the instance state/billing and terminate manually when appropriate.

After the 8-replicate short stage is complete, optionally run the long bridge on the same saved result directory:

```bash
SHARD=frankfurt STAGE=bridge MAX_HOURS=24 bash cloud/run_on_existing_instance.sh
# On the Ireland node:
SHARD=ireland STAGE=bridge MAX_HOURS=24 bash cloud/run_on_existing_instance.sh
```

## 3. Optional automated provisioning from your laptop (role-based)

**Prerequisites:** authenticated AWS CLI, S3 bucket, existing EC2 instance profile with the *narrow S3 permissions* shown in `cloud/iam_policy.example.json`, appropriate EC2 launch/IAM PassRole permissions, outbound Internet access on subnet, and at least 8 free G/VT On-Demand vCPUs in each selected region. For custom networking use each region's `FR_...` / `IE_...` overrides. Add `AmazonSSMManagedInstanceCore` to the role if Session Manager diagnostics are required; SSM-only permissions do not authorize S3 persistence.

First run a **read-only preflight** using your own current quotas; offerings do not guarantee live capacity:

```bash
export AWS_PROFILE=default
export FR_INSTANCE_TYPE=g6.2xlarge
export IE_INSTANCE_TYPE=g4dn.2xlarge
bash cloud/check_dual_region.sh
```

**No `aws sso login` is necessary** if the chosen profile already authenticates successfully. For EC2 launch, its caller permissions must allow RunInstances, SSM GetParameter, EC2 image inspection, S3 upload, and PassRole.

To provision **both** instances (this creates billable AWS resources):

```bash
export AWS_PROFILE=drpets
export RUN_ID=drpets-8rep-20261104
export S3_BUCKET=YOUR_EXISTING_BUCKET
export S3_REGION=YOUR_BUCKET_REGION
export INSTANCE_PROFILE=YOUR_EXISTING_EC2_S3_ROLE
export FR_INSTANCE_TYPE=g6.2xlarge
export IE_INSTANCE_TYPE=g4dn.2xlarge
export MODE=on-demand
export MAX_HOURS=24
export STAGE=development

bash cloud/launch_both_aws.sh
```

The launcher uploads the full project to S3, provisions one instance per region, uses the Deep Learning AMI, downloads source/checkpoints, installs dependencies, runs tests, executes the right shard, uploads files to S3 every 120 seconds and asks EC2 to terminate at normal completion. An independent on-instance watchdog initiates shutdown if the runtime limit is reached. The instance launch script was tested using mocked AWS responses only.

**If one launch succeeds and the other fails, the first remains billable.** The dual launcher isn't a transactional AWS operation. Use the printed instance IDs to monitor or terminate explicitly. A 24-hour limit applies **per instance**; two instances can consume 48 aggregate instance-hours per launcher invocation.

Same `RUN_ID` + same shard and stage resumes saved S3 results after a timed interruption. Do not reuse the RUN_ID for a changed code/config experiment; manifest mismatch intentionally refuses to resume.

For a bridge run, after all development shards have finished, set `STAGE=bridge` and use the **same RUN_ID** and launch again. Or run bridge on your existing machines using section 2.

## 4. Download & analyze the eight replicas

```bash
export RUN_ID=drpets-8rep-20261104
export S3_BUCKET=YOUR_EXISTING_BUCKET
export AWS_PROFILE=drpets
bash cloud/download_and_analyze.sh
```

This merges disjoint seed directories under `results/downloads/$RUN_ID/` and writes `eight_replica_analysis/replicate_effects.csv` and `summary.json` there. Override `DOWNLOAD_ROOT` if needed. Set `PYTHON` to your local analysis interpreter rather than the remote DLAMI path. Analysis treats **eight training replicas**, not 960 evaluation episodes, as eight independent observations. It refuses to compute a result unless all 8 replicas have 120 distinct PETS/DR-PETS episodes and saved manifests; also inspect validation completion, diagnostics and final exit statuses.

## 5. Safety and limitations

- An S3 bucket and IAM role are required **only for this unattended multi-day automatic EC2 deployment**. Do not put permanent AWS credentials in source code or EC2 user data.
- AWS Budgets alerts are **not** a hard spending cap. Review account quotas, On-Demand/Spot price, EBS charges, instance state and the log after launching.
- Availability of GPU families in a region does not imply permission, remaining quota, or available capacity.
- The full bridge/diagnostics can exceed earlier estimates depending on physics performance and actual GPU throughput.
- Physics mass/inertia updates call MuJoCo `mj_setConst`; official XML `settotalmass` warning is left unchanged to preserve the baseline model.
- `pynvml` is removed and `nvidia-ml-py` installed by `run_on_existing_instance.sh`.
- The Eq. (16) objective uses **neural-parameter Euclidean gradient norm**, not a general function-space Wasserstein guarantee.
