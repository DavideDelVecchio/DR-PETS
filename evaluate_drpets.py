import torch
from models.density_model import StateActionDensityModel

model = StateActionDensityModel(state_dim=4, action_dim=2)
model.flow.load_state_dict(torch.load("flow_density.pt"))
model.flow.eval()

sample_trajectory = torch.rand((10, 6))
log_likelihood = model.log_prob(sample_trajectory)
print("Log-likelihood:", log_likelihood)
