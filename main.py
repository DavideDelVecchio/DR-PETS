from models.density_model import StateActionDensityModel
from models.dynamics_ensemble import DynamicsEnsemble
from planners.cem_planner import CEMPlanner
from utils.config_loader import load_config, parse_cli_args
from utils.validate_config import validate_config
from utils.logging_hooks import log_planner_summary
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import gym

def run_planner_episode(planner):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = planner.plan(obs_tensor)
        action = int(action_tensor.squeeze().item())
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    return total_reward, steps

cli_args = parse_cli_args()
config = load_config("configs/config.yaml", cli_args)
validate_config(config)

planner_writer = None
if config['experiment']['planner_log_to_tensorboard']:
    planner_writer = SummaryWriter(log_dir=config['experiment']['planner_log_dir'])

model = StateActionDensityModel(
    state_dim=config['experiment']['state_dim'],
    action_dim=config['experiment']['action_dim']
)
model.flow.load_state_dict(torch.load(config['experiment']['density_model_path']))
density_model = model

ensemble_size = config['experiment']['ensemble_size']
dynamics_model = DynamicsEnsemble(
    state_dim=config['experiment']['state_dim'],
    action_dim=config['experiment']['action_dim'],
    ensemble_size=ensemble_size
)

# Load pretrained dynamics models from checkpoints
for i, model in enumerate(dynamics_model.models):
    path = f"checkpoints/dynamics_model_{i}.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:
        raise FileNotFoundError(f"Missing checkpoint: {path}")

dynamics_model.eval()

planner = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,
    lambda_penalty=config['experiment']['lambda_penalty'] if config['experiment']['use_drpets'] else 0.0,
    logger=planner_writer,
    action_dim=config['experiment']['action_dim']
)

for episode in range(100):
    reward, length = run_planner_episode(planner)
    if planner_writer:
        log_planner_summary(planner_writer, episode, reward, length)
