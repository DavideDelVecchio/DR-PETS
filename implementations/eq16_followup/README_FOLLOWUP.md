# DR-PETS follow-up study — ready for the November presentation

This is an **extension of the original eight-replica study**, using the **same, frozen** Equation (16) dynamics and planner code. No changes to the training or benchmark source files are necessary. New commands live in `followup.py` and `cloud/run_followup_existing_instance.sh`.

**Status:** renamed to `DRPETS_8replica_followup` and adapted for safe AWS follow-up execution. The complete repository-layout follow-up suite passes (68 tests and 30 subtests); reporting was regenerated from 960 original evaluation rows and checked for eight training replicas. Read-only verification of all eight real S3 checkpoints confirmed all 40 networks have three hidden layers. No new GPU experiment or AWS resource was launched by this adaptation.

`example_results/` contains eight small raw evaluation CSV fixtures, **not** trained weights. Local validation copies of real parent artifacts are under ignored `results/parent-results/` and must not be committed or packaged as source. To run `audit`, `bridge` or `curves`, supply each required `development_seed_<seed>/` with `ensemble.pt`, `run_manifest.json`, `config.json`, `validation.csv` and `raw.csv`.

## Architecture and frozen provenance

Each of the five Gaussian dynamics networks has **three 128-unit SiLU hidden
layers**, followed by a separate output head: `24 → 128 → 128 → 128 → 36`
(18 state + 6 action inputs, 18 means + 18 log standard deviations).
All eight real parent configurations specify `depth=3` and `smoke=false`.
The model preflight loads saved tensors rather than merely trusting defaults:

```bash
python cloud/verify_parent_models.py --shard frankfurt --stage audit --results-root results
```

`run_benchmark.py`, `eq16_core.py`, `tasks.py` and `drpets_benchmark.py` remain
byte-for-byte identical to the original deployed manifests. Do not replace them
with revised explanatory copies from another implementation. The repository
keeps this frozen workflow separately under `implementations/eq16_followup`;
the modern `implementations/eq16` and PETS-reference documentation remain intact.
Training smoke mode intentionally uses a smaller model and is not the study
architecture. Runtime package versions are recorded in logs; source hashes alone
do not certify identical Gymnasium/MuJoCo versions.

## Scientific scope

The original eight-replica short evaluation is unchanged. The follow-up adds:

1. **Audit:** integrity of checkpoint, five distinct finite networks, validation-computed epsilon selection, exactly 120 held-out paired episodes, and new held-out one-step NLL, RMSE, nominal transition prediction relative to no-motion, approximate marginal 90% interval coverage, and open-loop 1/5/15/30-step errors on random-action trajectories.
2. **Bridge:** two frozen representative ensembles, Frankfurt seed 101 and Ireland seed 113, each evaluated for nominal, low mass, high friction and weak motor at **1,000 steps**, five fresh paired episode seeds per condition, 40 episodes per replica, 80 total. The `--all-replicas` option expands this to eight ensembles (320 episodes) at extra cost. We intentionally use fresh bridge evaluation seeds (`seed+200003+97*k`) rather than the original short evaluation seeds.
3. **Curves:** eight additional physics settings across the same eight trained ensembles, each with 10 paired original held-out episode seeds and both controllers. Adds 1,280 episodes (mass 0.85, 1.15; friction 0.75, 1.25; actuator 0.50, 0.60, 0.80, 0.90). It **reuses the six original conditions**, including nominal and actuator 0.70, without rerunning them. The complete grids become mass (0.70, 0.85, 1.00, 1.15, 1.30), friction (0.50, 0.75, 1.00, 1.25, 1.50), actuator strength (0.50, 0.60, 0.70, 0.80, 0.90, 1.00).
4. **Analysis:** publication-style plots of **absolute reward** and **each method's shift-minus-own-nominal reward**. Shading is the 95% Student-t interval **across independent training replicas**, not across repeated evaluation episodes. Three-panel PDF, SVG, and PNG plots and machine-readable results are produced. Missing extension points are never invented or silently interpolated as observations.

The original planned eight independent training replicas are statistical units. Additional physics points and per-replica evaluation seeds are repeated measurements, **not** additional training replicates. These are development/exploratory experiments rather than independent confirmation.

## 1. Restore results on each existing GPU machine

Put the contents of this project on the existing EC2 machines and ensure your prior results are under the same `results/` directory:

```text
results/
  development_seed_101/
    ensemble.pt
    run_manifest.json
    config.json
    validation.csv
    raw.csv
  development_seed_103/ ...
```

Frankfurt uses seeds `101 103 107 109`; Ireland uses `113 127 131 137`. Do **not** substitute the bundled `example_results/` for the training-checkpoint tree; it only enables regenerating the current eight-replica plots.

Use the existing S3 backup if the files are no longer on local disk. Example on each instance:

```bash
aws s3 sync s3://YOUR-BUCKET/ORIGINAL-RUN/frankfurt/results/ results/ --only-show-errors
# On Ireland use its ireland/results prefix, not the Frankfurt prefix.
```

If your original S3 layout differs, adapt the URI. Do not force an unsafe `--resume` on unmatched checkpoint or code. The original manifest is verified, including code hashes and checkpoint SHA-256.

## 2. Run AUDIT first (no MPC evaluation needed)

```bash
python followup.py audit --shard frankfurt --device cuda --results-root results
python followup.py audit --shard ireland --device cuda --results-root results
```

Each seed writes `results/followup_audit_seed_<seed>/audit.json`. The audit checks a checkpoint is structurally valid and produces measured held-out dynamics metrics. **It cannot certify training convergence:** the original training did not persist per-epoch learning curves or optimizer state. A valid checkpoint is not a proof of a good control policy. Inspect one-step NLL, RMSE, model uncertainty, open-loop 1/5/15/30-step errors and the existing random-action benchmark. Source-code drift is deliberately a hard error.

## 3. Run the existing 1,000-step bridge

On the Frankfurt machine:

```bash
python followup.py bridge --shard frankfurt --device cuda \
  --workers 3 --results-root results
```

On Ireland:

```bash
python followup.py bridge --shard ireland --device cuda \
  --workers 3 --results-root results
```

Each command defaults to one representative seed in that region, producing `results/followup_bridge_seed_<seed>/raw.csv`. No retraining. The script uses the already-selected epsilon from `development_seed_<seed>/config.json`. It uses new evaluation seeds distinct from the short-episode evaluation. It resumes valid completed episodes and guards the exact configuration, checkpoint, physics, and source hashes.

To include all four replicas per region (substantially more GPU time): add `--all-replicas`.

## 4. Add eight physics points (incrementally, in parallel)

Start with the actuator curve, the least conclusive original perturbation:

```bash
python followup.py curves --shard frankfurt --family actuation --device cuda --workers 3 --results-root results
python followup.py curves --shard ireland --family actuation --device cuda --workers 3 --results-root results
```

Then run `--family mass` and `--family friction` separately, or `--family all` for all three. Each family has its own output directory and manifest, so it is safe to run or resume families independently. New physics points use **only one parameter changed at a time**, preserving nominal values for all others. No training is performed.

## 5. Use a single launcher on existing EC2 instances, with checkpoint sync

The wrapper defaults to `/opt/pytorch/bin/python`, installs all requirements, runs tests, checks live physics and saved model architecture, and executes only `followup.py`. It **does not provision or shut down EC2**. S3 parent inputs and follow-up outputs must be separate, non-overlapping prefixes; supply both URIs when using S3:

```bash
# Frankfurt example; use your bucket, parent run ID and NEW follow-up ID.
SHARD=frankfurt STAGE=audit MAX_HOURS=2 S3_REGION=BUCKET_REGION \
  S3_RESULTS_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/frankfurt/results \
  S3_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/followup/FOLLOWUP_ID/frankfurt/results \
  bash cloud/run_followup_existing_instance.sh

SHARD=frankfurt STAGE=bridge MAX_HOURS=24 S3_REGION=BUCKET_REGION \
  S3_RESULTS_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/frankfurt/results \
  S3_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/followup/FOLLOWUP_ID/frankfurt/results \
  bash cloud/run_followup_existing_instance.sh

SHARD=frankfurt STAGE=curves FAMILY=actuation MAX_HOURS=24 S3_REGION=BUCKET_REGION \
  S3_RESULTS_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/frankfurt/results \
  S3_URI=s3://BUCKET/drpets-eight/ORIGINAL_RUN/followup/FOLLOWUP_ID/frankfurt/results \
  bash cloud/run_followup_existing_instance.sh
```

Ireland uses its own original `ireland/results` prefix. Set `S3_RESULTS_REGION` if the parent bucket has a different region; `S3_REGION` refers to the output bucket, never compute placement. Logs default to a separate sibling prefix or explicit `S3_LOGS_URI`. Parent `development_seed_*` inputs are downloaded read-only; only `followup_*` directories are uploaded/restored in the extension namespace.

These extensions can take many GPU-hours. Setup/tests count against the same budget as execution; timeout normally returns 124 and failed transfers return nonzero. Resume requires the same follow-up ID, code and configuration; no automatic relaunch is performed. Credentials come from the EC2 role or configured local profile, not source code.

### 5a. Optional automated provisioning

`cloud/launch_followup_aws.sh` explicitly provisions one shard;
`cloud/launch_both_followup_aws.sh` preflights both regions and provisions two.
They default to **audit**, never development, retraining, bridge or all curves.
Source objects are checked before provisioning and duplicate active shard tags
block a launch. Live capacity can still fail after one node has been created.

Copy `cloud/followup.env.example` to ignored `cloud/followup.env`, or export
settings. The standalone working copy has local settings for the prior study
and a separate follow-up ID; only the generic example enters the repository.
Instance authentication uses a role, never the laptop profile from that file.

```bash
export AWS_PROFILE=default
export RUN_ID=ORIGINAL_STUDY_RUN_ID
export FOLLOWUP_ID=NEW_FOLLOWUP_ID
export S3_BUCKET=YOUR_STUDY_BUCKET
export S3_REGION=YOUR_BUCKET_REGION
export INSTANCE_PROFILE=YOUR_SCOPED_S3_SSM_PROFILE
# Read-only; offerings/quota do not reserve capacity:
bash cloud/check_dual_region.sh
# Also check both parent datasets and duplicate shard tags, without uploads/launch:
PREFLIGHT_ONLY=1 STAGE=audit bash cloud/launch_both_followup_aws.sh
# Billable only when explicitly requested:
SHARD=frankfurt STAGE=audit MAX_HOURS=2 bash cloud/launch_followup_aws.sh
# Alternatively, after approving both instance budgets:
STAGE=audit MAX_HOURS=2 bash cloud/launch_both_followup_aws.sh
```

Code is stored at
`drpets-eight/<RUN_ID>/followup/<FOLLOWUP_ID>/project.zip`.
Results/logs are beneath that extension's `<shard>/results/` and `<shard>/logs/`.
Original `project.zip`, parent checkpoints and CSVs are not overwritten.
The existing original-study scoped role covers this nested namespace;
`cloud/iam_policy.followup.example.json` provides a narrower parent-read/
extension-read-write template. No IAM changes are applied by these scripts.
A different bucket or prefix needs corresponding permissions.

Use regional `FR_...` / `IE_...` subnet, security-group and AMI overrides.
Custom parent roots use `FR_S3_RESULTS_URI` / `IE_S3_RESULTS_URI`, optionally
their corresponding `..._REGION` settings. Dual launch rejects global
region-specific IDs and ambiguous shared input/output URIs.

Automatic nodes request termination after completion; an independent watchdog
acts at the workload deadline plus five cleanup minutes. The default audit
budget is two hours **per instance**. Explicit bridge/curves examples use larger
windows and can consume many GPU-hours; two 24-hour nodes can use 48 aggregate
instance-hours plus cleanup grace. There is no automatic relaunch, implicit
bridge or hard dollar cap.

### 5b. Monitoring, recovery and validation

Inspect `/var/log/drpets/followup_bootstrap.log`, `nvidia-smi` and project logs
through Session Manager. Log/status names include stage/shard and curve family,
so separate families do not overwrite each other's records. Both result and
log uploads must succeed; no-data or absent exit markers are not completion.

Use the same follow-up ID and exact code/configuration to resume only the
affected shard after its old instance terminates. Manifest mismatch requires
investigation or a fresh follow-up ID, not deleting provenance guards.
Confirm actual instance termination and billing independently. Direct-on-
existing-instance wrappers do not shut the machine down. Do not open SSH to the
world or start two processes against the same shard/output namespace.

The local test evidence is under ignored `results/validation/`:
`followup-junit.xml`, `parent-architecture.json`, and `baseline-report/`.
Tests cover original scientific checks, grids/resume guards, three hidden
layers, safe source/output paths, stage dispatch, missing inputs, transfer/test
failures, deadlines and mocked dual-region provisioning. Local CPU wheel pins
are in `constraints-macos-intel-test.txt`; do not apply them to Linux/CUDA.
The original CSV fixtures were matched to S3 and report generation produced
960 episodes, eight replica effects and 12 condition/method summaries.
This is not a new live follow-up GPU performance result.

## 6. Rebuild actual plots and summaries

After collecting both regional results into one local `results/` tree:

```bash
python followup.py report --results-root results --output results/followup_analysis
```

The outputs are `physics_curves.pdf/png/svg`, `relative_to_nominal.pdf/png`, `episodes_all.csv`, `pairs.csv`, `per_replica_condition.csv`, `per_condition_ci.csv`, `replica_primary_effects.csv`, `summary.json`, and when present `bridge_paired.csv`, `bridge_per_replica_condition.csv`, `model_audit_summary.csv`.

To reproduce **today's original eight-replica plots now**, without having training checkpoints locally:

```bash
python followup.py report --results-root example_results --output baseline_eight_replica_figures
```

The package includes these baseline plots already rendered from the uploaded data. **Do not present the illustration-only generated images from previous sessions as exact numerical figures; these plots are generated deterministically from the evaluation CSVs.**

## Known limitations and checks before presentation

- The primary endpoint remains the eight original training-replica averages of DR-PETS minus matched PETS reward over five shifts. The baseline plots show **nominal plus original shifts**, with additional points only after the corresponding jobs complete.
- `PETS` is the **matched modern epsilon-zero controller**, not exact original Chua et al. PETS. Both planners share the same trained ensemble.
- Higher nominal reward than shifted reward improvement may imply **no demonstrated shift-specific robustness advantage**. Discuss the method-specific nominal-relative plots explicitly.
- Episodes are slow because MPC is replanned at each real environment step. The ~1.91× overhead measured previously is **evaluation execution time (MPC plus MuJoCo)**, *not* shared dynamics-model training time.
- One-step and open-loop predictions measured from random action sequences are valuable diagnostics but cannot by themselves establish sustained locomotion or model calibration under closed-loop trajectories.
- No new follow-up GPU jobs were executed by this adaptation. Real parent artifacts and all 40 network architectures were verified read-only, and CPU/offline tests passed. Follow-up CUDA execution, interruption recovery and live finalization require a separately authorized release check.
