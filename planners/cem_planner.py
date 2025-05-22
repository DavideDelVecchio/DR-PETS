import torch

def compute_dr_pets_score(predicted_states, actions, cost_fn, density_model, lambda_penalty):
    H, N, _ = actions.shape
    traj_cost = torch.zeros(N)
    sa_pairs = []
    for t in range(H):
        s = predicted_states[t]
        a = actions[t]
        traj_cost += cost_fn(s, a)
        sa_pairs.append(torch.cat([s, a], dim=-1))
    sa_all = torch.cat(sa_pairs, dim=0)
    log_probs = density_model.log_prob(sa_all).reshape(H, N)
    penalty = -log_probs.mean(dim=0)
    return traj_cost + lambda_penalty * penalty

class CEMPlanner:
    def __init__(self, dynamics_model, density_model=None, lambda_penalty=0.0, logger=None):
        self.dynamics_model = dynamics_model
        self.density_model = density_model
        self.lambda_penalty = lambda_penalty
        self.logger = logger

    def plan(self, obs_tensor):
        # Placeholder: Return a random action (0 or 1) for CartPole
        return torch.tensor([[0]]) if torch.rand(1).item() < 0.5 else torch.tensor([[1]])
