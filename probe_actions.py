import torch
from models.dynamics_ensemble import DynamicsEnsemble
from utils.config_loader import load_config
import torch.nn.functional as F

# Load config
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

# Load trained ensemble
ensemble = DynamicsEnsemble(state_dim, action_dim, ensemble_size)
for i, model in enumerate(ensemble.models):
    model.load_state_dict(torch.load(f"checkpoints/dynamics_model_{i}.pt"))
    model.eval()

# Pick a fixed state
state = torch.tensor([[0.0, 0.0, 0.05, 0.0]])  # slightly tilted pole

# Generate both possible one-hot actions for CartPole
actions = torch.eye(action_dim)

print("Probing ensemble predictions for different actions:\n")
for action in actions:
    action = action.unsqueeze(0).repeat(state.size(0), 1)
    predictions = torch.stack([model(state, action) for model in ensemble.models])
    mean_prediction = predictions.mean(dim=0)
    print(f"Action: {action.argmax().item()} → Predicted next state (mean across ensemble):\n{mean_prediction.squeeze().tolist()}\n")
