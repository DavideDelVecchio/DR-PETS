import torch

class CEMPlanner:
    """
    Cross-Entropy Method (CEM) Planner that supports:
      • Standard PETS scoring  (epsilon = 0)
      • DR-PETS scoring        (epsilon > 0, Wasserstein penalty)
    """

    def __init__(
        self,
        dynamics_model,
        density_model,
        epsilon,                # ← penalty weight (λ in DR-PETS paper)
        action_dim,
        horizon=30,
        num_samples=500,
        num_elites=50,
        num_iters=5,
        num_particles=10        # Q  (particles per sequence)
    ):
        self.dynamics_model = dynamics_model
        self.density_model   = density_model
        self.epsilon         = epsilon
        self.action_dim      = action_dim
        self.horizon         = horizon
        self.num_samples     = num_samples  # N
        self.num_elites      = num_elites
        self.num_iters       = num_iters
        self.num_particles   = num_particles  # Q

    # ------------------------------------------------------------------
    # PETS-compliant planning: sample N sequences, roll out Q particles,
    # update CEM distribution, return first action + PETS & DR-PETS scores
    # ------------------------------------------------------------------
    def plan(self, obs_tensor):
        device = obs_tensor.device
        T, N, Q = self.horizon, self.num_samples, self.num_particles

        # Initialise CEM distribution
        mean = torch.zeros(T, self.action_dim, device=device)
        std  = torch.ones (T, self.action_dim, device=device)

        for _ in range(self.num_iters):
            # 1) Sample N sequences  → clamp to action bounds [-1, 1]
            seq = torch.normal(
                mean.unsqueeze(1).expand(-1, N, -1),
                std .unsqueeze(1).expand(-1, N, -1)
            ).clamp(-1.0, 1.0)                              # [T, N, A]

            # 2) Broadcast to Q particles: [T, N, Q, A]
            seq = seq.unsqueeze(2).expand(-1, N, Q, -1)

            # Containers for reward and penalty
            rewards   = torch.zeros(N, Q, device=device)
            penalties = torch.zeros(N, Q, device=device) if self.epsilon > 0 else None

            # 3) Roll out each sequence on each ensemble model
            for model in self.dynamics_model.models:
                model.eval()
                # Duplicate the current state for every (N·Q) particle
                particles = obs_tensor.repeat(N * Q, 1).view(N, Q, -1)

                for t in range(T):
                    actions = seq[t]                                 # [N, Q, A]
                    sa_batch = torch.cat([particles, actions], dim=2).view(N * Q, -1)

                    # Predict next state
                    next_state = model(
                        particles.view(N * Q, -1),
                        actions  .view(N * Q, -1)
                    ).view(N, Q, -1)

                    # CartPole reward (+1 per time-step until termination)
                    rewards += 1.0 / len(self.dynamics_model.models)

                    # DR-PETS penalty
                    if self.epsilon > 0:
                        sa_batch = sa_batch.detach().clone().requires_grad_(True)
                        logp = self.density_model.flow.log_prob(sa_batch)
                        grad = torch.autograd.grad(logp.sum(), sa_batch, create_graph=False)[0]
                        grad_norm = grad.norm(dim=1).view(N, Q)
                        penalties += grad_norm / len(self.dynamics_model.models)

                    particles = next_state

            # 4) Aggregate over particles
            score_pets   = rewards.mean(dim=1)                        # [N]
            score_drpets = score_pets.clone()
            if self.epsilon > 0:
                score_drpets = score_pets - self.epsilon * penalties.mean(dim=1)

            # 5) Select elites and update mean / std
            elite_idx   = score_drpets.topk(self.num_elites, largest=True).indices
            elite_seqs  = seq[:, elite_idx, 0, :]                    # representative particle
            mean, std   = elite_seqs.mean(dim=1), elite_seqs.std(dim=1) + 1e-6

        # 6) Return the first action of the best sequence + diagnostic scores
        return mean[0].unsqueeze(0), {
            "score_pets":   score_pets  .mean().item(),
            "score_drpets": score_drpets.mean().item()
        }
