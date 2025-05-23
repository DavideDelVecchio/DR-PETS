import torch
from torch.utils.data import DataLoader, TensorDataset
from models.density_model import StateActionDensityModel
from utils.config_loader import load_config, parse_cli_args
from utils.validate_config import validate_config
from torch.utils.tensorboard import SummaryWriter

cli_args = parse_cli_args()
config = load_config("configs/config.yaml", cli_args)
validate_config(config)

writer = None
if config['experiment']['log_to_tensorboard']:
    writer = SummaryWriter(log_dir=config['experiment']['log_dir'])

data = torch.load(config['experiment']['state_action_dataset'])
sa_inputs = data[:, :config['experiment']['state_dim'] + config['experiment']['action_dim']]
dataset = TensorDataset(sa_inputs)
dataloader = DataLoader(dataset, batch_size=config['experiment']['batch_size'], shuffle=True)

model = StateActionDensityModel(
    state_dim=config['experiment']['state_dim'],
    action_dim=config['experiment']['action_dim']
)
model.fit(dataloader, epochs=config['experiment']['num_epochs'], lr=1e-3)
torch.save(model.flow.state_dict(), config['experiment']['density_model_path'])
if writer:
    writer.close()