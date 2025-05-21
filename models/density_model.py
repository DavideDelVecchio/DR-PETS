import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

class StateActionDensityModel:
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_flows=5):
        self.input_dim = state_dim + action_dim
        self.flow = self._build_flow(hidden_dim, num_flows)

    def _build_flow(self, hidden_dim, num_flows):
        transforms = [
            MaskedAffineAutoregressiveTransform(self.input_dim, hidden_dim)
            for _ in range(num_flows)
        ]
        transform = CompositeTransform(transforms)
        distribution = StandardNormal([self.input_dim])
        return Flow(transform, distribution)

    def fit(self, data_loader, epochs=20, lr=1e-3):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        self.flow.train()
        for epoch in range(epochs):
            for batch in data_loader:
                x = batch[0]
                loss = -self.flow.log_prob(x).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def log_prob(self, state_action):
        self.flow.eval()
        return self.flow.log_prob(state_action)
