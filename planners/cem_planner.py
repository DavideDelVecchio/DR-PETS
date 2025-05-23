import torch
import torch.nn.functional as F

def compute_dr_pets_score(predicted_states, actions, cost_fn, density_model, lambda_penalty):
    H, N, d = actions.shape
    traj_cost = torch.zeros(N)
    sa_pairs = []
    penalty = torch.zeros_like(traj_cost)
    for t in range(H):
        s = predicted_states[t]
        a = actions[t]
        traj_cost += cost_fn(s, a)
        sa_pairs.append(torch.cat([s, a], dim=-1))

    sa_all = torch.cat(sa_pairs, dim=0)
    if lambda_penalty > 0.0:
        sa_all = sa_all.clone().detach().requires_grad_(True)
        log_probs = density_model.log_prob(sa_all)
        grad_logp = torch.autograd.grad(log_probs.sum(), sa_all, create_graph=False)[0]
        grad_logp = grad_logp.view(H, N, -1)

        weights = traj_cost.view(1, N, 1).expand(H, N, grad_logp.size(-1))
        weighted_grad = grad_logp * weights
        penalty_vec = weighted_grad.sum(dim=0)  # [N, d]
        penalty = penalty_vec.norm(dim=1)  # [N]

    return traj_cost + lambda_penalty * penalty

class CEMPlanner:
    def __init__(self, dynamics_model, density_model=None, lambda_penalty=0.0, logger=None,
                 horizon=15, num_samples=500, num_elites=50, num_iters=5, action_dim=2):
        self.dynamics_model = dynamics_model
        self.density_model = density_model
        self.lambda_penalty = lambda_penalty
        self.logger = logger
        self.H = horizon
        self.N = num_samples
        self.K = num_elites
        self.I = num_iters
        self.action_dim = action_dim
        self.action_low = 0
        self.action_high = 1

    def plan(self, obs):
        mean = torch.full((self.H, 1), 0.5)
        std = torch.full((self.H, 1), 0.3)

        for _ in range(self.I):
            samples = torch.normal(
                mean.unsqueeze(1).expand(-1, self.N, -1),
                std.unsqueeze(1).expand(-1, self.N, -1)
            ).clamp(0, self.action_dim - 1).round()

            predicted_states = [obs.expand(self.N, -1)]
            for t in range(self.H):
                next_states = []
                for model in self.dynamics_model.models:
                    actions_onehot = F.one_hot(
                        samples[t].long().view(-1), num_classes=self.action_dim
                    ).float()
                    ns = model(predicted_states[-1], actions_onehot)
                    next_states.append(ns)
                mean_next_state = torch.stack(next_states).mean(dim=0)
                predicted_states.append(mean_next_state)

            predicted_states = predicted_states[:-1]

            def cost_fn(s, a):
                x = s[:, 0]
                theta = s[:, 2]
                return x.abs() + theta.abs()

            actions_onehot_seq = torch.stack([
                F.one_hot(samples[t].long().view(-1), num_classes=self.action_dim).float()
                for t in range(self.H)
            ])

            scores = compute_dr_pets_score(
                predicted_states, actions_onehot_seq, cost_fn,
                self.density_model, self.lambda_penalty
            )
            elite_idxs = scores.topk(self.K, largest=False).indices
            elite_samples = samples[:, elite_idxs, :]
            mean = elite_samples.mean(dim=1)
            std = elite_samples.std(dim=1) + 1e-2

        action_plan = mean.round()
        print("Planned action sequence:", action_plan.squeeze().tolist())
        return action_plan[0].unsqueeze(0)
