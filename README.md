# DR-PETS (Distributionally-Robust PETS)

Implementation of **DR-PETS**  
([paper](https://arxiv.org/abs/2503.20660)) with:

* Probabilistic Ensembles with Trajectory Sampling (PETS)
* Wasserstein regularizer (ε) for distributional robustness
* Online dynamics + density retraining loop

## Quick Start

```bash
git clone https://github.com/<YOU>/DR-PETS.git
cd DR-PETS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Collect random data, then train+plan online
python scripts/main_train_and_plan.py --config configs/cartpole.yaml
