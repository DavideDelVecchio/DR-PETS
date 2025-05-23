import torch
import torch.nn as nn

class StateActionDensityModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, num_flows=5):
        super().__init__()
        from nflows.flows import Flow
        from nflows.distributions.normal import StandardNormal
        from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

        self.features = state_dim + action_dim

        # Create 5 masked autoregressive transforms to match the saved model
        transforms = []
        for _ in range(num_flows):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.features,
                    hidden_features=hidden_dim,
                    num_blocks=2,
                    use_residual_blocks=False,
                    random_mask=False,
                    activation=nn.ReLU(),
                    dropout_probability=0.0
                )
            )

        # Combine transforms
        transform = CompositeTransform(transforms)
        
        # Create flow with standard normal base distribution
        self.flow = Flow(transform, StandardNormal([self.features]))

    def log_prob(self, x):
        # Clone if x doesn't require gradients, otherwise use as-is
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        return self.flow.log_prob(x)

    def fit(self, data, epochs=50, batch_size=128, lr=1e-3, device='cpu'):
        self.flow.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for (batch,) in loader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(batch).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.size(0)
            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
