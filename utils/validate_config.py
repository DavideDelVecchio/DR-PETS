def validate_config(config):
    required_keys = [
        'use_drpets', 'lambda_penalty', 'state_dim', 'action_dim',
        'batch_size', 'density_model_path', 'state_action_dataset',
        'num_epochs', 'log_to_tensorboard', 'log_dir',
        'planner_log_to_tensorboard', 'planner_log_dir'
    ]
    for key in required_keys:
        if key not in config['experiment']:
            raise ValueError(f"Missing required config key: {key}")