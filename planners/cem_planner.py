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
