import torch
import numpy as np
from models.density_model import StateActionDensityModel
from utils.config_loader import load_config

# Load config and parameters
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
dataset_path = config['experiment']['density_dataset_path']  # path to tensor file
save_path = config['experiment']['density_model_path']       # path to save model

# Load dataset
print("Loading dataset from:", dataset_path)
data = torch.load(dataset_path)  # shape [N, state_dim + action_dim + next_state_dim]
print(f"Original dataset shape: {data.shape}")

# Extract only state and action parts (first state_dim + action_dim features)
data = data[:, :state_dim + action_dim]
print(f"Using only state+action features: {data.shape}")

# Initialize and train model
model = StateActionDensityModel(state_dim, action_dim)
model.fit(data, epochs=50, batch_size=128, device='cuda' if torch.cuda.is_available() else 'cpu')

# Save model
print("Saving trained density model to:", save_path)
torch.save(model.flow.state_dict(), save_path)
