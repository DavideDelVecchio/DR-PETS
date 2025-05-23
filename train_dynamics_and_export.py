import argparse
import torch
from models.dynamics_ensemble import DynamicsEnsemble
from utils.config_loader import load_config

# Parse device argument
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help='Device to train on (e.g., "cuda" or "cpu")')
args = parser.parse_args()
device = args.device

# Load config and dataset
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

data = torch.load("state_action_dataset.pt").to(device)
states = data[:, :state_dim]
actions = data[:, state_dim:state_dim + action_dim]
next_states = data[:, state_dim + action_dim:]
targets = next_states

# Train ensemble
ensemble = DynamicsEnsemble(state_dim, action_dim, ensemble_size)
for i, model in enumerate(ensemble.models):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        model.train()
        pred = model(states, actions)
        loss = ((pred - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Model {i} | Epoch {epoch+1} | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), f"checkpoints/dynamics_model_{i}.pt")
