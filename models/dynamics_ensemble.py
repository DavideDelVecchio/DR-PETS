import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DynamicsEnsemble:
    def __init__(self, state_dim, action_dim, ensemble_size):
        self.models = [DynamicsModel(state_dim, action_dim) for _ in range(ensemble_size)]

    def to(self, device):
        for model in self.models:
            model.to(device)
        return self

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def predict(self, state, action):
        return [model(state, action) for model in self.models]
