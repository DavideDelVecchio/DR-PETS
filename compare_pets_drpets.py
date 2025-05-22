import torch
from utils.config_loader import load_config, parse_cli_args
from utils.validate_config import validate_config
from utils.logging_hooks import log_evaluation_metrics
from torch.utils.tensorboard import SummaryWriter

from models.density_model import StateActionDensityModel
from models.dynamics_ensemble import DynamicsEnsemble
from planners.cem_planner import CEMPlanner

import gym

def make_env():
    return gym.make("CartPole-v1")

def run_planner_episode(planner):
    env = make_env()
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # Select best action from CEM planner
        action_tensor = planner.plan(obs_tensor)
        action = int(action_tensor.squeeze().item())
        obs, reward, terminated,truncated, _ = env.step(action)
        done = terminated or truncated 
        total_reward += reward
        steps += 1
    return total_reward, steps

cli_args = parse_cli_args()
config = load_config("configs/config.yaml", cli_args)
validate_config(config)

def evaluate(planner, label, writer=None):
    all_rewards = []
    for episode in range(10):
        reward, length = run_planner_episode(planner)
        all_rewards.append(reward)
        if writer:
            log_evaluation_metrics(writer, {f"{label}/reward": reward}, episode)
    return all_rewards

# Load shared components
env = make_env()
ensemble_size = config['experiment']['ensemble_size']
dynamics_model = DynamicsEnsemble(
    state_dim=config['experiment']['state_dim'],
    action_dim=config['experiment']['action_dim'],
    ensemble_size=ensemble_size
)

# DR-PETS setup
density_model = StateActionDensityModel(
    state_dim=config['experiment']['state_dim'],
    action_dim=config['experiment']['action_dim']
)
density_model.flow.load_state_dict(torch.load(config['experiment']['density_model_path']))

writer = SummaryWriter(log_dir="runs/compare")

planner_drpets = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,
    lambda_penalty=config['experiment']['lambda_penalty'],
    logger=writer
)

drpets_rewards = evaluate(planner_drpets, label="DRPETS", writer=writer)

# PETS baseline (lambda = 0)
planner_pets = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,  # used but not penalized
    lambda_penalty=0.0,
    logger=writer
)

pets_rewards = evaluate(planner_pets, label="PETS", writer=writer)

print("Avg DR-PETS Reward:", sum(drpets_rewards) / len(drpets_rewards))
print("Avg PETS Reward:", sum(pets_rewards) / len(pets_rewards))

# Optional benchmark comparison
benchmark_score = config['experiment'].get('benchmark_score', 500.0)  # CartPole-v1 benchmark
if benchmark_score is not None:
    dr_gap = benchmark_score - (sum(drpets_rewards) / len(drpets_rewards))
    pets_gap = benchmark_score - (sum(pets_rewards) / len(pets_rewards))
    print("Gap to benchmark (DR-PETS):", dr_gap)
    print("Gap to benchmark (PETS):", pets_gap)
    writer.add_scalar("Benchmark/DRPETS_gap", dr_gap)
    writer.add_scalar("Benchmark/PETS_gap", pets_gap)

writer.close()
