from models.density_model import StateActionDensityModel
from models.dynamics_ensemble import DynamicsEnsemble
from utils.config_loader import load_config, parse_cli_args
from utils.validate_config import validate_config
from utils.logging_hooks import log_planner_summary
from torch.utils.tensorboard import SummaryWriter

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

planner = CEMPlanner(...,
                     dynamics_model=dynamics_model,
                     density_model=density_model,
                     lambda_penalty=config['experiment']['lambda_penalty']
                     if config['experiment']['use_drpets'] else 0.0,
                     logger=planner_writer)

for episode in range(100):
    reward, length = run_planner_episode(planner)
    if planner_writer:
        log_planner_summary(planner_writer, episode, reward, length)
