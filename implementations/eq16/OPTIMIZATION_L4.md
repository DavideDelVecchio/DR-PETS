# L4 / g6.2xlarge performance optimization

This implementation continues to evaluate Equation (16), the Monte-Carlo
reward-weighted transition-density *parameter* score. No scientific objective
has been changed. No code from any GitHub branch was used.

## Why the previous instance used one vCPU

- `run_benchmark.py` previously evaluated validation and test episodes serially.
- Each CEM decision synchronizes GPU -> CPU to select elites (`scores.cpu().numpy()`).
- `eq16_core.py` runs Python loops over ensemble members and horizon steps,
  inside small candidate chunks (`chunk=8`). GPU memory footprint can be tiny.
- `torch.set_num_threads(4)` does **not** parallelize Python loops or sequential
  Gymnasium environments; the GIL prevents parallel execution of Python code
  within threads.
- 270 MiB L4 memory is not evidence of an OOM or memory bottleneck.

## Safe changes implemented

1. `--workers 3` or `--workers 4` evaluates independent episodes in
   `ProcessPoolExecutor` processes, using **spawn** to safely initialize CUDA.
   Processes share the physical L4 but each has its own CUDA context and
   its own once-loaded copy of the same trained ensemble.
2. `--worker-threads 1` prevents 3-4 workers from oversubscribing all CPU cores.
3. `--chunk 16` rather than 8 is a useful *starting point* for larger GPU
   microbatches. Its speed is not established on an L4; test 8,16,32 if needed.
4. Online PETS data-collection episodes are parallelized as well.
5. Validation and held-out episode results are appended as each completes.
6. `--load-ensemble` enables benchmarking an existing `ensemble.pt` without
   rerunning initial training or the 10,750-transition online collection.

WARNING: Independent workers use memory for multiple CUDA contexts and submit
kernels to one GPU. More workers are not guaranteed to improve throughput;
start with 3, then measure 1,3,4. On CUDA OOM or poor speed, reduce to 2.

## How to run the optimized benchmark on an existing EC2 instance

The previously running process cannot be patched in place. It can finish
normally, or you can stop it after securing a usable checkpoint. You can
re-launch with this package, or transfer the updated files to the instance
before starting a *new* process. For the no-role launcher, previous model
checkpoints become available after its result tarball has been uploaded.

If the prior run has already archived `results/cloud/ensemble.pt`, extract
it locally and set `RESUME_ENSEMBLE=/path/to/ensemble.pt` when invoking
`cloud/launch_aws_no_instance_role.sh`. The launcher embeds only that model
checkpoint in the new zip and appends `--load-ensemble resume/ensemble.pt`
to the new run. This skips training, including on-policy data collection.

```bash
python -m pytest -q tests

# Resume *evaluation*, using the SAME previously trained model,
# without repeating any training/online data collection.
python run_benchmark.py --task halfcheetah --device cuda \
  --load-ensemble results/cloud/ensemble.pt \
  --ensemble 5 --width 128 --depth 3 \
  --workers 3 --worker-threads 1 --threads 2 \
  --chunk 16 --horizon 15 --particles 16 --population 64 \
  --cem-iterations 3 --epsilon-grid 0 .0001 .001 \
  --val-seeds 3 --eval-seeds 3 --eval-steps 150 \
  --out results/parallel_pilot
```

`--ensemble`, `--width` and `--depth` MUST match the saved model. If they do
not match, `load_state_dict` fails instead of silently using incorrect weights.

For targeted timing without retraining:

```bash
bash cloud/probe_l4.sh results/cloud/ensemble.pt
```

This probe compares workers=1/chunk=8, workers=1/chunk=16,
workers=3/chunk=16, workers=4/chunk=16 on a shorter task.
**Caution:** it is a benchmark, not a zero-cost operation. It executes many
MPC decisions (by default 5 steps × 6 shifts × 2 controllers per setting).

## Deploy on AWS

`cloud/launch_aws_no_instance_role.sh` is the no-custom-IAM launcher previously
used, updated for:

- default `MODE=on-demand` (Spot may be unavailable in Frankfurt);
- `RUN_ARGS` default includes `--workers 3 --worker-threads 1 --chunk 16`;
- validation tests run as a suite.

Start it with your existing `AWS_PROFILE`, `S3_BUCKET`, and network settings
using the README instructions. The launcher provisions a *new* EC2 instance.
It cannot update a previously running instance automatically. Review EC2
instance termination and billing state after each run.

## Profiling

Use these commands on EC2:

```bash
nvidia-smi dmon -s u -d 1
ps -eo pid,pcpu,pmem,args --sort=-pcpu | head -15
```

Interpretation: if 3 workers each use ~1 core and GPU utilization increases
while wall time decreases, the optimization is helping. If utilization
stays low, next steps are CEM vectorization, fewer Python-level synchronizations,
and batched/fused ensemble computations. If GPU utilization saturates, adding
more workers is unlikely to help.

## Scientific validity

Multiprocessing uses the identical trained checkpoint for every episode;
paired seed values and per-episode random number generation are unchanged.
A CPU regression test verifies equal serial vs parallel raw rewards. The CUDA
multi-process path has not been executed here (no CUDA available).
