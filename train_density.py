import argparse
import torch
import numpy as np
from models.density_model import StateActionDensityModel
from utils.config_loader import load_config

# Parse device argument
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu', help='Device to train on (e.g., "cuda" or "cpu")')
args = parser.parse_args()
device = args.device

# Load config and parameters
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
dataset_path = config['experiment']['density_dataset_path']
save_path = config['experiment']['density_model_path']


# Load dataset
print("Loading dataset from:", dataset_path)
data = torch.load(dataset_path).to(device)
data = data[:, :state_dim + action_dim]
# Initialize and train model
model = StateActionDensityModel(state_dim, action_dim)
model.fit(data, epochs=50, batch_size=128, device=device)

# Save model
print("Saving trained density model to:", save_path)
torch.save(model.flow.state_dict(), save_path)
