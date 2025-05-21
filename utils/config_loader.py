import yaml
import argparse

def load_config(path, cli_args=None):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if cli_args:
        for k, v in cli_args.items():
            if k in config['experiment']:
                config['experiment'][k] = v
    return config


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_penalty', type=float)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--planner_log_dir', type=str)
    parser.add_argument('--use_drpets', type=bool)
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}