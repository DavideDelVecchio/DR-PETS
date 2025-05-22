
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

3. **Compare PETS vs DR-PETS**

```bash
python scripts/compare_pets_drpets.py
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

## 📬 Credits

Based on PETS (Chua et al. 2018) and extensions from DR-PETS research.

---

## 🔍 TODO

- Add support for continuous control (e.g., HalfCheetah)
- Integrate video recording for episode rollouts
- Extend DR-cost to include epistemic uncertainty tracking
