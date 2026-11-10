# Diagnosed T4 pilot and resumable validation

The supplied pilot-results.tar.gz completed 24 of 45 validation jobs and zero held-out jobs.
PETS episodes averaged 229 seconds; positive-epsilon Equation 16 episodes averaged
557 seconds per episode with three concurrent workers. `MAX_HOURS=1` is too short.

## Resume without retraining or replaying the completed 24 jobs

In the directory containing the extracted new project, run:

```bash
export AWS_REGION=eu-central-1
export AWS_PROFILE=drpets
export INSTANCE_TYPE=g4dn.2xlarge   # change to the actual T4 instance size you used
export MODE=on-demand
export S3_BUCKET=your-existing-bucket
export RESUME_ARCHIVE=/absolute/path/to/pilot-results.tar.gz
export MAX_HOURS=5
export PRESIGN_SECONDS=21600
bash cloud/launch_aws_no_instance_role.sh
```

This script packs the previous `ensemble.pt`, `validation.csv`, and optionally
`raw.csv` into the new instance's project. `run_benchmark.py --resume` verifies
that existing rows belong to the requested seed/physics/radius configuration
and executes only missing episodes. The old archive is preserved.

CAUTION: This is a NO-INSTANCE-ROLE launcher. S3 presigned URLs may expire early
when the AWS identity that signed them uses short-lived SSO/CLI credentials.
If a run is likely to exceed the credential lifetime, use the role-based launcher
or run smaller independent chunks with persisted results; do not assume an
increased PRESIGN_SECONDS extends temporary credentials.

CAVEAT: We have not launched the revised script against a live EC2 instance.

## Additional observations

* The archive does NOT contain a `DRPETS_EXIT_CODE=` marker, even though the
  prior no-role launcher should have appended it. A timeout is highly likely
  from duration and episode timings, but exact shutdown cause remains unproved.
* The XML `settotalmass` deprecation and `pynvml` warning did not cause an error.
* Selecting the best epsilon or comparing against PETS requires complete
  validation and held-out samples; current partial data are insufficient.
