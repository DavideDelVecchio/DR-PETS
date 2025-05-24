
This repository implements PETS and DR-PETS for evaluating model-based planning performance, benchmarked on the CartPole-v1 environment.
The  DR-PETS algorithm is described in <https://arxiv.org/abs/2503.20660>
---

## 🛠️ Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📦 Requirements

Dependencies include:

- `torch`
- `nflows`
- `gym`
- `matplotlib`
- `PyYAML`
- `tensorboard`

---

## 🚀 Workflow

1. **Collect state-action data**

```bash
python collect_dataset.py
```

2. **Train the density model**

```bash
python train_density.py
```
1. **Train the dynamics**

```bash
python train_dynamics_and_export.py
```


4. **Compare PETS vs DR-PETS**

```bash
python compare_drpets_pets_different_pole_length.py
```

4. **Visualize results**

```bash
tensorboard --logdir runs/compare
```

---

## 📁 Configuration

Edit `configs/config.yaml` to set hyperparameters and paths:

- `ensemble_size`: number of models in the dynamics ensemble
- `state_dim`, `action_dim`: dimensions of your environment
- `lambda_penalty`: DR-PETS robustness weight
- `benchmark_score`: reference reward for comparison (e.g. 500.0 for CartPole)

---

## 🧪 Benchmark

Results are evaluated against the CartPole-v1 benchmark score of 500.

---

## 🧠 DR-PETS Robustness Objective

Unlike PETS, which maximizes expected reward over an ensemble of dynamics models, **DR-PETS directly optimizes a regularized reward objective**. It adds a penalty to trajectories that pass through low-density, uncertain regions of the state-action space:

DR-PETS Objective:

    J_DR = ExpectedReward - lambda * UncertaintyPenalty

The penalty term is computed as the negative log-likelihood of the state-action sequence under a learned density model. This:

- Eliminates the need for explicit perturbations or adversarial sampling.
- Encourages the planner to stay in well-modeled, high-likelihood regions.
- Makes the policy robust to model errors by avoiding out-of-distribution behavior.
  
## 📬 Credits

Based on PETS (Chua et al. 2018) [text](https://arxiv.org/abs/1805.12114) and extensions from DR-PETS  <https://arxiv.org/abs/2503.20660>

---

## 🔍 TODO

- Add support for continuous control (e.g., HalfCheetah)
- Integrate video recording for episode rollouts
- Extend DR-cost to include epistemic uncertainty tracking
