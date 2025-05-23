import torch
import gym
from models.density_model import StateActionDensityModel
from models.dynamics_ensemble import DynamicsEnsemble
from planners.cem_planner import CEMPlanner
from utils.config_loader import load_config

config = load_config("configs/config.yaml")
state_dim = config['experiment']['state_dim']
action_dim = config['experiment']['action_dim']
ensemble_size = config['experiment']['ensemble_size']

# Load models
density_model = StateActionDensityModel(state_dim, action_dim)
density_model.flow.load_state_dict(torch.load(config['experiment']['density_model_path']))
density_model.flow.eval()

dynamics_model = DynamicsEnsemble(state_dim, action_dim, ensemble_size)
for i, model in enumerate(dynamics_model.models):
    model.load_state_dict(torch.load(f"checkpoints/dynamics_model_{i}.pt"))
    model.eval()

# Initialize planner
planner = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,
    lambda_penalty=0.0,
    logger=None,
    horizon=30,
    num_samples=1000,
    num_elites=100,
    num_iters=7,
    action_dim=action_dim
)

# Rollout debug
env = gym.make("CartPole-v1")
obs, _ = env.reset()
obs_traj = []
action_traj = []
total_reward = 0

done = False
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_tensor = planner.plan(obs_tensor)
    action = int(action_tensor.squeeze().item())
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    obs_traj.append(obs.tolist())
    action_traj.append(action)
    total_reward += reward

print("=== PETS DEBUG ROLLOUT ===")
print(f"Total reward: {total_reward}")
print("Trajectory length:", len(obs_traj))
print("Actions:", action_traj)
print("Final observation:", obs_traj[-1])

# DR-PETS comparison rollout
planner_drpets = CEMPlanner(
    dynamics_model=dynamics_model,
    density_model=density_model,
    lambda_penalty=config['experiment']['lambda_penalty'],
    logger=None,
    horizon=30,
    num_samples=1000,
    num_elites=100,
    num_iters=7,
    action_dim=action_dim
)
env = gym.make("CartPole-v1")
obs, _ = env.reset()
obs_traj = []
action_traj = []
total_reward = 0

obs, _ = env.reset()
obs_traj_drpets = []
action_traj_drpets = []
total_reward_drpets = 0
done = False
while not done:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_tensor = planner_drpets.plan(obs_tensor)
    action = int(action_tensor.squeeze().item())
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    obs_traj_drpets.append(obs.tolist())
    action_traj_drpets.append(action)
    total_reward_drpets += reward

print("=== DR-PETS DEBUG ROLLOUT ===")
print(f"Total reward: {total_reward_drpets}")
print("Trajectory length:", len(obs_traj_drpets))
print("Actions:", action_traj_drpets)
print("Final observation:", obs_traj_drpets[-1])
