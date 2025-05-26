"""
State-Action Density Model (normalizing flow) for DR-PETS.

Uses nflows (https://github.com/bayesiains/nflows).
Trains to maximise log-likelihood of collected (state, action) pairs.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from nflows.flows      import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base          import CompositeTransform


class StateActionDensityModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, num_layers: int = 4):
        super().__init__()
        sa_dim = state_dim + action_dim

        transforms = []
        for _ in range(num_layers):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=sa_dim,
                    hidden_features=hidden,
                    num_blocks=2,
                    use_residual_blocks=False
                )
            )

        transform = CompositeTransform(transforms)
        base_dist = StandardNormal([sa_dim])
        self.flow = Flow(transform, base_dist)
        
        # Convert all parameters to float32
        self.to(torch.float32)

    # Convenience
    def log_prob(self, sa: torch.Tensor) -> torch.Tensor:
        sa = sa.to(torch.float32)
        return self.flow.log_prob(sa)

    # Simple trainer (used by main_train_and_plan.py)
    def fit(self, data: torch.Tensor, epochs: int = 50, batch_size: int = 128, lr: float = 1e-3, device=None):
        if device:
            self.to(device)
            data = data.to(device)
        data = data.to(dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        for _ in range(epochs):
            for batch in loader:
                loss = -self.log_prob(batch).mean()
                opt.zero_grad(); loss.backward(); opt.step()
