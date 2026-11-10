# Code-level comparison: `epsilon=0` baseline versus Chua et al.'s PETS (2018)

**Scope.** This is a static source-code comparison against the original PETS reference repository, [`kchua/handful-of-trials`](https://github.com/kchua/handful-of-trials), at commit [`77fd8802cc30b7683f0227c90527b5414c0df34c`](https://github.com/kchua/handful-of-trials/tree/77fd8802cc30b7683f0227c90527b5414c0df34c), versus the `implementations/eq16/` package distributed in this repository migration. This is *not* a head-to-head numerical reproduction of the original PETS paper.

## Exact sense in which `epsilon=0` recovers PETS

Our robust CEM candidate score is

\[
\widehat{S}_\epsilon(u)=\frac1B\sum_{b=1}^{B}\frac1Q\sum_{q=1}^{Q} \sum_{t=0}^{H-1}\gamma^t r_t^{b,q}(u)-\epsilon \sqrt{\frac1B\sum_{b=1}^{B}\|\widehat g_b(u)\|_2^2}.
\]

At `epsilon=0`, our `eq16_scores()` returns the ensemble- and particle-averaged simulated return with **no sensitivity penalty**. No gradient penalty is computed, as demonstrated by the `epsilon>0` branches in [`eq16_core.py`](../implementations/eq16/eq16_core.py). The dynamics training, trajectory propagation and CEM are **our implementation's own design**. Thus the correct statement is: `epsilon=0` **recovers this project's unregularized PETS-style controller**, **not** the original 2018 PETS code, full methodology or published learning curves.

**Parameter-name collision:** The original PETS CEM has a parameter named `epsilon` that is the *CEM variance stopping threshold* (`max(var) > epsilon`). It is **not** a Wasserstein radius or DR penalty. See [original `cem.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/optimizers/cem.py#L15-L32) and [CEM termination](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/optimizers/cem.py#L69-L93).

## Audited implementation differences

| Component | Original PETS reference (`handful-of-trials`) | Our `epsilon=0` baseline | Consequence |
|---|---|---|---|
| Probabilistic model | Five learned diagonal-Gaussian neural networks (`PE` by default), each predicting a mean and variance; bounded, **trainable** min/max log-variance with softplus; NLL and weight decay | Five PyTorch `GaussianDynamics` networks predicting normalized delta mean and **clamped log-standard-deviation** (`[-5,1]`), NLL on normalized targets, no equivalent per-layer weight-decay recipe | Same broad Gaussian ensemble idea, different predictive densities and calibration |
| HalfCheetah NN architecture | Four hidden layers, **200 Swish units** each, custom preprocessing and 18-output dynamics | Three hidden layers, **128 SiLU units** each, 18-dimensional state delta and a 36-output mean/log-SD head | Changes representational capacity, training speed and statistical estimator |
| HalfCheetah observation and targets | Custom 18D observation begins with **forward velocity**; training applies trig preprocessing to a joint angle and predicts *next forward velocity* plus **deltas for other state components** | Gymnasium HalfCheetah-v5 with `exclude_current_positions_from_observation=False`: observation includes **absolute root x**; predicts the delta of all 18 observation coordinates | Different Markov representation, target distribution and out-of-distribution behavior; nominal rewards may have related physical meaning but benchmarks are not identical |
| Dynamics training | Independent per-member bootstrap indices; input standardization; default HalfCheetah **5 training epochs per online update**; continue training the same model across 300 experiment iterations | Independent bootstrap indices and input + target standardization; typically **25 epochs**, initial random transition pool and far fewer online collection rounds; **reinitialize ensemble** for each online retraining round | Different data distribution, update dynamics, total environment interactions and optimization trajectory |
| Particle propagation | `TSinf` by default: each trajectory particle stays assigned to one model; default **20 particles total**, constrained divisible by ensemble size, i.e. **4 per network** for five models | Fixed model per particle trajectory (TS∞-style); **16 particles per model**, hence **80 total** across five models | Same TS∞-style model assignment, but **four times as many total particles** in the example; different simulation budget |
| CEM for HalfCheetah | Planning horizon **30**, **500 candidates**, **50 elites**, max **5 iterations**, `alpha=0.1`; bounded **truncated Gaussian**, variance-based early stopping | Horizon **15**, **64 candidates**, approx. **9 elites** (15% rounded down), **3 iterations**, previous-mean weight `0.25`; unbounded Gaussian followed by action clipping; fixed iteration count | Different optimizer, computational budget and possibly action ranking; 0.1 and 0.25 are smoothing weights, not DR radii |
| MPC warm start | Reuses and **shifts prior optimized action sequence** to initialize the next call | Reinitializes the CEM Gaussian distribution afresh at every control step | Important difference in convergence and receding-horizon behavior |
| Return aggregation | Sum of undiscounted per-step observation/action **costs**, averaged across 20 particles | Sum of per-step reward with **`gamma=0.99`** (default), averaged across 80 particles | Even for identical dynamics, may select different actions due to discounting |
| HalfCheetah costs | `obs_cost_fn = -obs[:,0]` (predicted forward velocity); `ac_cost_fn = 0.1 * sum(a²)` | `(x' - x)/dt - 0.1 * sum(a²)` reward (Gymnasium state contains root position) | Related physical running objective, but different state encoding/environment; check exact XML/dt/reward semantics before direct score comparison |
| Environment and episode protocol | Custom `MBRLHalfCheetah-v0` XML, MuJoCo 1.x; task horizon **1000**; **300** training iterations of one rollout each by default | Gymnasium `HalfCheetah-v5`, modern native MuJoCo; 150-step development evaluation, optional 1000-step bridge; physics shifts | Original and modern environment benchmarks are not interchangeable |
| CartPole / Pendulum | Original **custom MuJoCo CartPole** with continuous actions; no Pendulum experiment in original reference tree | Lightweight local CartPole with discrete actions and a lightweight Pendulum | Neither toy task is an exact reference PETS environment reproduction |
| Frameworks | TensorFlow 1, Gym 0.9.4, `mujoco-py`, historical NumPy/GPflow | PyTorch 2.x, Gymnasium/MuJoCo | Maintenance modernization; **not sufficient by itself** to explain the algorithmic differences above |

### Exact code locations

**Original PETS** (pinned commit):

- [`dmbrl/config/halfcheetah.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/config/halfcheetah.py): horizon, planning population, model architecture, observation preprocessing, targets and reward/cost.
- [`dmbrl/config/default.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/config/default.py): default `PE`, `TSinf`, five model members and 20 total particles.
- [`dmbrl/controllers/MPC.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/controllers/MPC.py): particle grouping, trajectory propagation, warm-started MPC, observation and action costs.
- [`dmbrl/misc/optimizers/cem.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/optimizers/cem.py): CEM truncated samples, variance threshold, elite update and smoothing.
- [`dmbrl/modeling/models/BNN.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/modeling/models/BNN.py): trainable variance bounds, Gaussian NLL, bootstrap learning, input scaler.
- [`dmbrl/env/half_cheetah.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/env/half_cheetah.py): velocity-first observation, action costs, custom XML and reward.
- [`dmbrl/misc/MBExp.py`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/MBExp.py): online collection, model retraining and evaluation protocol.
- [`requirements.txt`](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/requirements.txt): historical dependency pins.

**Current independent implementation (this package):**

- [`eq16_core.py`](../implementations/eq16/eq16_core.py): lines 21–59 Gaussian dynamics/sampling, 62–104 robust objective and `epsilon=0`, 123–144 training.
- [`run_benchmark.py`](../implementations/eq16/run_benchmark.py): lines 50–74 candidate generation/CEM, 275–307 training/online collection, 415–443 defaults.
- [`tasks.py`](../implementations/eq16/tasks.py): HalfCheetah-v5 state, reward and physics shifts.

## What to claim

**Supported:** "Setting the DR-PETS regularization coefficient to zero recovers the unregularized, probabilistic-ensemble MPC controller implemented in this repository. Both controllers share identical learned transition models, objective and optimizer configurations, isolating the addition of the return-sensitivity penalty."

**Not supported:** "Our `epsilon=0` controller reproduces the original PETS algorithm's experimental implementation or Chua et al.'s published HalfCheetah scores."

For comparison against *reference PETS*, separately reproduce/match the original benchmark: align simulator/XML, observation and target preprocessing, reward timing, per-step horizon, model architecture/variance constraints, training data schedule, TS∞ particle count, CEM and warm start. A two-baseline experiment is scientifically stronger: (A) our matched `epsilon=0` baseline for causal penalty ablation, and (B) an adapted reference-PETS baseline for absolute competitive performance. Verify both on identical **modern** simulator conditions before attributing any differences to algorithm choice.
