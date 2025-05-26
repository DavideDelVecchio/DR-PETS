import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim, dtype=torch.float32)
        )

    def forward(self, state, action):
        x = torch.cat([state.to(dtype=torch.float32), 
                      action.to(dtype=torch.float32)], dim=-1)
        return self.net(x)

class DynamicsEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, ensemble_size):
        super().__init__()
        self.models = nn.ModuleList([DynamicsModel(state_dim, action_dim) 
                                   for _ in range(ensemble_size)])

    def forward(self, state, action):
        predictions = [model(state, action) for model in self.models]
        return torch.stack(predictions).mean(dim=0)

    def predict(self, state, action):
        return [model(state, action) for model in self.models]
