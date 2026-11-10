# Implementations and branches

## Historical state-action density heuristic

Branch: `legacy/state-action-density-2025` (based on the pre-migration `master` commit). The old CartPole code uses a deterministic dynamics neural ensemble and a separately trained normalizing-flow state-action density `q_φ(s,a)`; its planner penalizes the **input score** `||∇_(s,a) log q_φ(s,a)||`. This is a behavioral/data-distribution heuristic, **not the return sensitivity** `∇_θ E[J]` in the DR-PETS Equation (16)/(21) derivation. It also adds a constant reward during simulated planning, which cannot rank actions for fixed-horizon CartPole nominally. Its old README describes it as DR-PETS, but the archived README clarifies that these implementations are distinct.

Historical branches `main`, `full-implementation`, `full-metal-device`, `surrogate-rewards` remain available without rewriting their history. They should not be labeled equivalent to current Equation (16) experiments.

## Current Equation (16) research implementation

Folder: `implementations/eq16/`.

- `eq16_core.py`: `GaussianDynamics` predicts normalized mean/log-standard-deviation of a **next-state delta**, trained by Gaussian negative log likelihood with independent bootstrap sampling. `sample_and_score` samples a next state and evaluates the transition score at the **detached** sample. `eq16_scores` estimates return and reward-to-go-weighted parameter gradients across particles and ensemble members; it takes the RMS of **per-model gradient estimates**, subtracts `ε` times that value from nominal expected return, and leaves weights frozen during planning.
- `run_benchmark.py`: model-based CEM action planning, nominal training/online collection, validation-grid epsilon selection, paired held-out evaluation, structured results and resume manifests.
- `drpets_benchmark.py`: lightweight CartPole/Pendulum tasks (convenient for smoke tests, **not exact official Gymnasium physics**).
- `tasks.py`: environment adapters and official Gymnasium HalfCheetah-v5 physics shifts; updated MuJoCo constants following body-mass/inertia changes (`mj_setConst`).
- `run_shard.py`, `analyze_shards.py`, `study_design.json`: eight independent replicate partitioning, controlled physics shifts, matched paired-seed comparison and per-replicate aggregation.
- `cloud/`: region preflight, optional EC2 launch, S3 checkpoint/resume and performance probes. The role-based unattended launch requires existing AWS permissions/roles; `legacy/` contains earlier single-instance prototypes.
- `tests/`: tests for score estimator on Gaussian toy systems, epsilon=0 identity, physics correction, job reuse and independent seed assignment.

### PETS benchmark baseline

Setting `epsilon=0` disables the gradient penalty in **the same scorer** and leaves all other CEM and learned-model settings matched. This is intentional: it estimates the effect of adding Equation (16), not a head-to-head comparison with the historical PETS research repository.

### Cautions

- `W2` is instantiated using an *NN-parameter Euclidean surrogate*, not a proof that the function-space ambiguity ball contains the true physical dynamics.
- A finite reference ensemble and finite-particle gradient norm introduce modeling and estimation error.
- Ensemble member selection uses a fixed model within each trajectory (TS-infinity-style) rather than every trajectory-sampling variant in the original PETS work.
- HalfCheetah XML's deprecated `settotalmass` attribute belongs to Gymnasium; the XML is preserved to maintain the nominal model.
