import torch
import sys
import gym
import matplotlib.pyplot as plt
from planners.cem_planner import CEMPlanner
from models.dynamics_ensemble import DynamicsEnsemble
from models.density_model import StateActionDensityModel
from utils.config_loader import load_config

# Check GPU availability
if not torch.cuda.is_available():
    print("ERROR: CUDA GPU not available. Please run on a GPU-enabled Azure VM like Standard_NC6s_v3.")
    sys.exit(1)

# Load configuration
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

def load_models():
    dynamics_model = DynamicsEnsemble(state_dim, action_dim, ensemble_size)
    for i, model in enumerate(dynamics_model.models):
        model.load_state_dict(torch.load(f"checkpoints/dynamics_model_{i}.pt"))
        model.eval()

    density_model = StateActionDensityModel(state_dim, action_dim)
    density_model.flow.load_state_dict(torch.load(config['experiment']['density_model_path']))
    density_model.flow.eval()
    return dynamics_model, density_model

def evaluate(pole_lengths, lambda_penalty, num_episodes=50):
    dynamics_model, density_model = load_models()
    results = []

    for l in pole_lengths:
        rewards = []

        for episode in range(num_episodes):
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
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tensor = planner.plan(obs_tensor)
                action = int(action_tensor.item())
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

        avg_reward = sum(rewards) / num_episodes
        results.append((l, avg_reward))
        print(f"Pole length: {l:.2f} -> Avg total reward over {num_episodes} episodes: {avg_reward:.2f}")

    return results

# Run evaluations
lengths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
rewards_pets = evaluate(lengths, lambda_penalty=0.0)
rewards_drpets = evaluate(lengths, lambda_penalty=config['experiment']['lambda_penalty'])

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(*zip(*rewards_pets), label="PETS", marker='o')
plt.plot(*zip(*rewards_drpets), label="DR-PETS", marker='x')
plt.xlabel("Pole Length (m)")
plt.ylabel("Average Total Reward")
plt.title("PETS vs DR-PETS under Pole Length Variations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/results/pole_length_comparison.png")
print("Saved plot to outputs/results/pole_length_comparison.png")
