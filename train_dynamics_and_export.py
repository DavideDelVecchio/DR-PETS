import torch
from models.dynamics_ensemble import DynamicsEnsemble
from utils.config_loader import load_config, parse_cli_args
from utils.validate_config import validate_config
from torch.utils.data import TensorDataset, DataLoader
import os

cli_args = parse_cli_args()
config = load_config("configs/config.yaml", cli_args)
validate_config(config)

# Load dataset (expects saved tuples of (state, action, next_state))
data = torch.load(config['experiment']['state_action_dataset'])
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

states = data[:, :state_dim]
actions = data[:, state_dim:state_dim + action_dim]
next_states = data[:, state_dim + action_dim:]
targets = next_states



dataset = TensorDataset(states[:-1], actions[:-1], targets[:-1])
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

ensemble = DynamicsEnsemble(state_dim, action_dim, ensemble_size)
optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in ensemble.models]

for epoch in range(20):
    for x_s, x_a, x_next in dataloader:
        for i, model in enumerate(ensemble.models):
            optimizers[i].zero_grad()
            pred = model(x_s, x_a)
            loss = torch.nn.functional.mse_loss(pred, x_next)
            loss.backward()
            optimizers[i].step()

os.makedirs("checkpoints", exist_ok=True)
for i, model in enumerate(ensemble.models):
    torch.save(model.state_dict(), f"checkpoints/dynamics_model_{i}.pt")
print("Dynamics ensemble training complete and saved.")
