import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from planners.cem_planner import CEMPlanner
from models.dynamics_ensemble import DynamicsEnsemble
from models.density_model import StateActionDensityModel
from utils.config_loader import load_config

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load configuration
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

def load_models():
    dynamics_model = DynamicsEnsemble(state_dim, action_dim, ensemble_size).to(device)
    for i, model in enumerate(dynamics_model.models):
        model.load_state_dict(torch.load(f"checkpoints/dynamics_model_{i}.pt", map_location=device))
        model.eval()

    density_model = StateActionDensityModel(state_dim, action_dim).to(device)
    density_model.flow.load_state_dict(torch.load(config['experiment']['density_model_path'], map_location=device))
    density_model.flow.eval()
    return dynamics_model, density_model

def evaluate(pole_lengths, lambda_penalty, num_episodes=50):
    dynamics_model, density_model = load_models()
    results = []

    for l in pole_lengths:
        rewards = []

        for _ in range(num_episodes):
            env = gym.make("CartPole-v1")
            env.env.length = l

            planner = CEMPlanner(
                dynamics_model=dynamics_model,
                density_model=density_model,
                lambda_penalty=lambda_penalty,
                action_dim=action_dim,
                horizon=15,
                num_samples=500,
                num_elites=50,
                num_iters=5
            )

            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = planner.plan(obs_tensor)
                action = int(action_tensor.item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        stderr = np.std(rewards, ddof=1) / np.sqrt(num_episodes)
        results.append((l, avg_reward, stderr))
        print(f"Pole length: {l:.2f} -> Avg reward: {avg_reward:.2f} ± {1.96 * stderr:.2f} (95% CI)")

    return results

# Run evaluations
lengths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
rewards_pets = evaluate(lengths, lambda_penalty=0.0)
rewards_drpets = evaluate(lengths, lambda_penalty=config['experiment']['lambda_penalty'])

# Extract for plotting
lengths_p, means_p, stderr_p = zip(*rewards_pets)
lengths_d, means_d, stderr_d = zip(*rewards_drpets)

# Plot results with 95% confidence intervals
plt.figure(figsize=(8, 5))
plt.errorbar(lengths_p, means_p, yerr=1.96 * np.array(stderr_p), label="PETS", marker='o', capsize=5)
plt.errorbar(lengths_d, means_d, yerr=1.96 * np.array(stderr_d), label="DR-PETS", marker='x', capsize=5)
plt.xlabel("Pole Length (m)")
plt.ylabel("Average Total Reward")
plt.title("PETS vs DR-PETS under Pole Length Variations (95% CI, GPU)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/results/pole_length_comparison_gpu_ci.png")
print("Saved plot with confidence intervals to outputs/results/pole_length_comparison_gpu_ci.png")
