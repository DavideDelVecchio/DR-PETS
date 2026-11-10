# DR-PETS — implementations and reproducible control benchmarks

**Research code for distributionally robust model-predictive control with probabilistic neural dynamics.** This repository maintains distinct, explicitly documented implementations. They must not be conflated with one another or with the upstream PETS reference code.

## Start here

| Implementation | Location | Dynamics / robustness mechanism | Status |
|---|---|---|---|
| **Equation (16) / (21) DR-PETS (current)** | [`implementations/eq16/`](implementations/eq16/) | Bootstrap-trained Gaussian transition ensembles; Monte Carlo return-to-go-weighted transition log-likelihood scores with respect to **network parameters**; RMS gradient penalty; CEM | Research implementation; locally unit-tested; empirical robustness only |
| **PETS baseline (current)** | Same implementation, `epsilon=0` | Uses precisely the same learned Gaussian ensemble, reward and CEM evaluator, with the regularizer disabled | Matched experimental baseline, **not** an exact reproduction of Chua et al. (2018) |
| **Frozen eight-replica follow-up** | [`implementations/eq16_followup/`](implementations/eq16_followup/) | Exact parent-evaluator hashes; saved three-hidden-layer ensembles; audit, long bridge and new physics curves without retraining | Separate frozen snapshot for original-checkpoint compatibility; [guide](implementations/eq16_followup/README_FOLLOWUP.md) |
| **Historical state-action density heuristic** | [`legacy/state-action-density-2025`](https://github.com/DavideDelVecchio/DR-PETS/tree/legacy/state-action-density-2025) (branch) | Deterministic next-state ensemble with a separate normalizing-flow density score `∇_(s,a) log q(s,a)` | Archived alternative approach; **not Equation (16)** |
| Other experiments | Existing branches `main`, `full-implementation`, `full-metal-device`, `surrogate-rewards` | Historic exploratory implementations; see branch-specific README | Retained for provenance; not the primary supported benchmark |

## Supported tasks

| Task | Environment | Action space | Notes |
|---|---|---|---|
| **CartPole** | Self-contained lightweight dynamics (`drpets_benchmark.py`) | Discrete | Fast correctness/smoke experiment, *not* Gymnasium CartPole-v1 equivalence |
| **Pendulum** | Self-contained lightweight dynamics (`drpets_benchmark.py`) | Continuous | Small continuous-action control test, *not* Gymnasium Pendulum-v1 equivalence |
| **HalfCheetah** | **Gymnasium HalfCheetah-v5 / MuJoCo** (`tasks.py`) | Continuous (6D) | Primary physics-shift benchmark; 18D observations retain root X so forward-velocity rewards can be calculated in model rollouts |

## Quick start

Python 3.10+ and PyTorch are required. HalfCheetah requires a functional MuJoCo/Gymnasium installation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r implementations/eq16/requirements.txt
cd implementations/eq16
python -m pytest -q tests
python run_benchmark.py --task cartpole --smoke --device cpu
python run_benchmark.py --task pendulum --smoke --device cpu
# GPU + MuJoCo:
python run_benchmark.py --task halfcheetah --smoke --device cuda
```

For the eight independent training replicates, seed sharding, AWS deployment, resumable evaluation, and analysis, start with [the experiment guide](implementations/eq16/README_SPLIT_AWS.md) and [the study plan](implementations/eq16/STUDY_PLAN.md). **No AWS infrastructure is created by import or tests.** Provisioning requires an explicit call to the AWS scripts.

For follow-ups on the original saved ensembles, use
[`implementations/eq16_followup`](implementations/eq16_followup/README_FOLLOWUP.md),
not a newly trained model. It reuses the original 960 evaluation rows and adds
only requested audit/bridge/curve jobs. Its frozen source is intentionally
separate from the modern implementation: even explanatory source edits change
the hashes used by the original manifests. All 40 saved networks were checked
read-only: five models per replica, three hidden layers of width 128, and a
separate Gaussian output head. `make followup-report` regenerates plots from the
small committed CSV fixtures without checkpoints, training, MPC or AWS writes.
Generated figures, actual weights and private AWS settings remain ignored.

## Important scientific qualifications

1. **Equation (16) is implemented numerically under a Euclidean NN-parameter-space interpretation**: `g_b = E[Σ_t G_t ∇_(θ_b) log f_(θ_b)(s_(t+1)|s_t,a_t)]`. A different function-space Wasserstein ground metric gives a different dual norm. The code is *not* a certificate for the original paper's unspecified model-function metric.
2. The Monte Carlo **norm of an estimated gradient is biased** at finite particle count. Increase particles and assess estimator variance before attributing robustness gains to the mathematical penalty.
3. This repository's current PETS is a **modern matched baseline**, not the exact PETS research code from Chua et al. The upstream implementation uses a legacy TensorFlow 1 / GPflow / OpenAI Gym / `mujoco-py` stack; the present implementation deliberately uses PyTorch, Gymnasium and native MuJoCo. Other architecture and rollout choices also differ. See the [file-by-file PETS reference trace](docs/PETS_REFERENCE_DIFFERENCES.md), including model/observation differences, particle counts, CEM warm start and the distinct meanings of `epsilon`.
4. The initial HalfCheetah physics-shift experiments are **exploratory**, and multiple episodes from one learned ensemble are not independent training replications.

## Documentation

- [Implementation comparison](docs/IMPLEMENTATIONS.md)
- [PETS baseline versus the original reference implementation](docs/PETS_REFERENCE_DIFFERENCES.md)
- [Equation (16), assumptions, limitations](docs/EQUATION16.md)
- [Eight-replica experimental protocol](implementations/eq16/STUDY_PLAN.md)
- [AWS Frankfurt + Ireland deployment](implementations/eq16/README_SPLIT_AWS.md)
- [Migration and tags](docs/MIGRATION.md)
- [Frozen follow-up workflow and safe AWS namespace](implementations/eq16_followup/README_FOLLOWUP.md)

## Papers and attribution

- Jesawada et al., **DR-PETS: Learning-Based Control With Planning in Adversarial Environments**, 2025: https://arxiv.org/abs/2503.20660
- Chua et al., **Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models**, NeurIPS 2018: https://arxiv.org/abs/1805.12114
- Original PETS experiment code: https://github.com/kchua/handful-of-trials

This is independent research code, not an official implementation endorsed by either paper's authors. The historical contribution is preserved in its own branch and tagged commit rather than erased.
