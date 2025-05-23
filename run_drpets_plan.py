import argparse
import torch
import gym
from models.dynamics_ensemble import DynamicsEnsemble
from models.density_model import StateActionDensityModel
from planners.cem_planner import CEMPlanner
from utils.config_loader import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_size', type=int, default=5)
parser.add_argument('--particles', type=int, default=10)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()

device = args.device

# Load config and environment
config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
density_path = config['experiment']['density_model_path']

# Load models
density_model = StateActionDensityModel(state_dim, action_dim)
density_model.flow.load_state_dict(torch.load(density_path, map_location=device))
density_model.flow.eval()

dynamics_model = DynamicsEnsemble(state_dim, action_dim, args.ensemble_size)
for i, model in enumerate(dynamics_model.models):
    model.load_state_dict(torch.load(f"checkpoints/dynamics_model_{i}.pt", map_location=device))
    model.eval()

planner = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,
    lambda_penalty=config['experiment']['lambda_penalty'],
    horizon=15,
    num_samples=args.particles * 50,
    num_elites=args.particles * 5,
    num_iters=5,
    action_dim=action_dim
)

env = gym.make("CartPole-v1")

perturbed_envs = {}
for m in [0.5, 0.75, 1.0, 1.25, 1.5]:
    env_m = gym.make("CartPole-v1")
    env_m.env.masscart = m
    perturbed_envs[f"mass_{m:.1f}kg"] = env_m
total_rewards = []
for ep in range(args.episodes):
    obs, _ = env.reset()
    ep_reward = 0
    for t in range(args.steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = planner.plan(obs_tensor)
        action = int(action_tensor.item())
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(ep_reward)
    print(f"Episode {ep+1}: Reward = {ep_reward}")

print("=== DR-PETS Planning Summary ===")
print(f"Base environment average reward: {sum(total_rewards)/len(total_rewards):.2f}")

for label, penv in perturbed_envs.items():
    perturbed_rewards = []
    for ep in range(5):
        obs, _ = penv.reset()
        ep_reward = 0
        for t in range(args.steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_tensor = planner.plan(obs_tensor)
            action = int(action_tensor.item())
            obs, reward, terminated, truncated, _ = penv.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        perturbed_rewards.append(ep_reward)
    avg_perturbed = sum(perturbed_rewards) / len(perturbed_rewards)
    print(f"{label} average reward: {avg_perturbed:.2f}")
print(f"Average Reward over {args.episodes} episodes: {sum(total_rewards)/len(total_rewards):.2f}")

import matplotlib.pyplot as plt

labels = []
avg_rewards = []
for label, penv in perturbed_envs.items():
    labels.append(label)
    avg_rewards.append(sum([
        sum(
            planner.plan(torch.tensor(penv.reset()[0], dtype=torch.float32).unsqueeze(0)).item()
            for _ in range(args.steps)
        ) for _ in range(5)
    ]) / 5)

plt.figure(figsize=(10, 5))
plt.bar(labels, avg_rewards, color='skyblue')
plt.xticks(rotation=45)
plt.ylabel("Average Reward")
plt.title("DR-PETS Performance under Mass Perturbations")
plt.tight_layout()
plt.savefig("drpets_mass_robustness.png")
print("Plot saved as drpets_mass_robustness.png")
