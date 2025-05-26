#!/usr/bin/env python3
"""
Azure ML Workspace Deployment Script for DR-PETS Project

This script:
1. Connects to Azure ML workspace
2. Configures compute targets with GPU support
3. Sets up the training environment with dependencies
4. Submits and monitors training jobs
5. Handles experiment tracking and logging
"""

import os
import sys
import argparse
import logging
import yaml
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

# Azure ML imports
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration
from azureml.widgets import RunDetails
from azureml.exceptions import WorkspaceException, UserErrorException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('azure_deploy.log')
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'azure': {
        'subscription_id': None,  # Must be provided or in environment
        'resource_group': None,   # Must be provided or in environment
        'workspace_name': 'drpets-workspace',
        'region': 'eastus',
        'compute_name': 'gpu-cluster',
        'vm_size': 'Standard_NC6',  # GPU VM with 6 cores
        'min_nodes': 0,
        'max_nodes': 4,
        'idle_seconds_before_scaledown': 1200,  # 20 minutes
    },
    'experiment': {
        'name': 'dr-pets-training',
        'description': 'DR-PETS training with GPU acceleration',
        'state_dim': 4,
        'action_dim': 2,
        'ensemble_size': 5,
        'epsilon': 0.05,
        'episodes': 100,
        'budget_limit_usd': 50.0,  # Maximum budget in USD
        'checkpointing': {
            'frequency': 10,  # episodes
            'enabled': True,
        },
        'tensorboard': {
            'enabled': True,
        }
    }
}

class AzureMLDeployer:
    def __init__(self, config_path=None):
        """Initialize the deployer with configuration."""
        self.config = DEFAULT_CONFIG.copy()
        
        # Try to load config from file
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Update config with user values
                    if 'azure' in user_config:
                        self.config['azure'].update(user_config['azure'])
                    if 'experiment' in user_config:
                        self.config['experiment'].update(user_config['experiment'])
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                sys.exit(1)
        
        # Check for required Azure configuration in environment variables
        if not self.config['azure']['subscription_id']:
            self.config['azure']['subscription_id'] = os.environ.get('AZURE_SUBSCRIPTION_ID')
            if not self.config['azure']['subscription_id']:
                logger.error("Azure subscription ID not found in config or environment variables (AZURE_SUBSCRIPTION_ID)")
                sys.exit(1)
                
        if not self.config['azure']['resource_group']:
            self.config['azure']['resource_group'] = os.environ.get('AZURE_RESOURCE_GROUP')
            if not self.config['azure']['resource_group']:
                logger.error("Azure resource group not found in config or environment variables (AZURE_RESOURCE_GROUP)")
                sys.exit(1)
        
        self.workspace = None
        self.compute_target = None
        self.environment = None
        self.experiment = None
    
    def connect_workspace(self):
        """Connect to Azure ML workspace, creating it if necessary."""
        try:
            # Try to connect to existing workspace
            logger.info(f"Connecting to workspace {self.config['azure']['workspace_name']}...")
            self.workspace = Workspace.get(
                name=self.config['azure']['workspace_name'],
                subscription_id=self.config['azure']['subscription_id'],
                resource_group=self.config['azure']['resource_group']
            )
            logger.info("Successfully connected to existing workspace")
            return True
        except WorkspaceException:
            # Workspace doesn't exist, create it
            try:
                logger.info(f"Workspace not found. Creating new workspace {self.config['azure']['workspace_name']}...")
                self.workspace = Workspace.create(
                    name=self.config['azure']['workspace_name'],
                    subscription_id=self.config['azure']['subscription_id'],
                    resource_group=self.config['azure']['resource_group'],
                    location=self.config['azure']['region'],
                    create_resource_group=True,
                    exist_ok=True
                )
                logger.info(f"Workspace created successfully in {self.config['azure']['region']}")
                return True
            except Exception as e:
                logger.error(f"Failed to create workspace: {e}")
                return False
    
    def setup_compute_target(self):
        """Set up GPU compute target for training."""
        if not self.workspace:
            logger.error("No workspace available. Connect to workspace first.")
            return False
            
        compute_name = self.config['azure']['compute_name']
        
        try:
            # Check if compute target already exists
            self.compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
            logger.info(f"Using existing compute target: {compute_name}")
        except ComputeTargetException:
            # Create a new compute target
            try:
                logger.info(f"Creating new compute target: {compute_name}...")
                
                # Configure compute cluster
                config = AmlCompute.provisioning_configuration(
                    vm_size=self.config['azure']['vm_size'],
                    min_nodes=self.config['azure']['min_nodes'],
                    max_nodes=self.config['azure']['max_nodes'],
                    idle_seconds_before_scaledown=self.config['azure']['idle_seconds_before_scaledown'],
                    vm_priority='dedicated'  # Use 'lowpriority' for cheaper but preemptible VMs
                )
                
                # Create the compute target
                self.compute_target = ComputeTarget.create(self.workspace, compute_name, config)
                self.compute_target.wait_for_completion(show_output=True)
                
                logger.info(f"Compute target '{compute_name}' created successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to create compute target: {e}")
                return False
        
        return True
    
    def setup_environment(self):
        """Set up the Azure ML environment with required dependencies."""
        if not self.workspace:
            logger.error("No workspace available. Connect to workspace first.")
            return False
            
        try:
            # Create a new environment
            env_name = f"drpets-env-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            logger.info(f"Creating environment: {env_name}...")
            
            # Start with curated PyTorch environment for GPU support
            self.environment = Environment.get(workspace=self.workspace, name="AzureML-PyTorch-1.7-GPU")
            
            # Add custom dependencies
            conda_deps = CondaDependencies()
            
            # Add packages from requirements.txt
            project_dir = Path(__file__).resolve().parent.parent
            requirements_path = project_dir / "requirements.txt"
            
            if requirements_path.exists():
                with open(requirements_path, 'r') as f:
                    for line in f:
                        package = line.strip()
                        if package and not package.startswith('#'):
                            if '@' in package:  # Handle git URLs specially
                                # For nflows from GitHub
                                if 'nflows' in package:
                                    conda_deps.add_pip_package("git+https://github.com/bayesiains/nflows.git")
                            else:
                                conda_deps.add_pip_package(package)
            else:
                # Fallback dependencies if requirements.txt not found
                conda_deps.add_pip_package("torch==2.1.0")
                conda_deps.add_pip_package("gym==0.26.2")
                conda_deps.add_pip_package("numpy")
                conda_deps.add_pip_package("matplotlib")
                conda_deps.add_pip_package("pyyaml")
                conda_deps.add_pip_package("tensorboard")
                conda_deps.add_pip_package("tqdm")
                conda_deps.add_pip_package("git+https://github.com/bayesiains/nflows.git")
            
            # Add dependencies for Azure ML
            conda_deps.add_pip_package("azureml-core")
            conda_deps.add_pip_package("azureml-mlflow")
            
            self.environment.python.conda_dependencies = conda_deps
            
            # Register the environment
            self.environment.register(workspace=self.workspace)
            
            logger.info(f"Environment '{env_name}' created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set up environment: {e}")
            return False
    
    def prepare_training_script(self):
        """Prepare the entry script for Azure ML training."""
        try:
            # Create a wrapper script that sets up Azure ML logging
            script_path = Path(__file__).resolve().parent / "azure_training_wrapper.py"
            
            with open(script_path, 'w') as f:
                f.write("""#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path

# Azure ML imports
from azureml.core import Run

# Get the current run context
run = Run.get_context()

def main():
    parser = argparse.ArgumentParser(description='DR-PETS training wrapper for Azure ML')
    parser.add_argument('--config', type=str, default='azure_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Log parameters to Azure ML
    for key, value in config['experiment'].items():
        if isinstance(value, (int, float, str, bool)):
            run.log(f"config_{key}", value)
    
    # Set up TensorBoard logging
    if config['experiment'].get('tensorboard', {}).get('enabled', False):
        os.environ['TENSORBOARD_LOGS_DIR'] = './logs'
        os.makedirs('./logs', exist_ok=True)
    
    # Run the main training script
    import main_train_and_plan
    
    # Log final metrics
    if os.path.exists('./results.csv'):
        run.upload_file('results.csv', './results.csv')
    
    # Upload checkpoints
    checkpoint_dir = Path('./checkpoints')
    if checkpoint_dir.exists():
        for checkpoint_file in checkpoint_dir.glob('*.pt'):
            run.upload_file(f'checkpoints/{checkpoint_file.name}', str(checkpoint_file))
    
    run.complete()

if __name__ == "__main__":
    main()
""")
            
            logger.info(f"Created training wrapper script at {script_path}")
            return script_path
        except Exception as e:
            logger.error(f"Failed to prepare training script: {e}")
            return None
    
    def prepare_azure_config(self):
        """Create Azure-specific configuration file."""
        try:
            # Create a config file for Azure ML training
            config_path = Path(__file__).resolve().parent / "azure_config.yaml"
            
            # Start with a copy of the experiment config
            azure_config = {
                'experiment': self.config['experiment'].copy()
            }
            
            # Add Azure-specific settings
            azure_config['experiment']['device'] = 'cuda'
            azure_config['experiment']['checkpointing']['path'] = './checkpoints'
            
            with open(config_path, 'w') as f:
                yaml.dump(azure_config, f)
            
            logger.info(f"Created Azure configuration at {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"Failed to prepare Azure config: {e}")
            return None
    
    def submit_training_job(self):
        """Submit the training job to Azure ML."""
        if not all([self.workspace, self.compute_target, self.environment]):
            logger.error("Workspace, compute target, and environment must be set up first.")
            return False
            
        try:
            # Create experiment
            experiment_name = self.config['experiment']['name']
            self.experiment = Experiment(workspace=self.workspace, name=experiment_name)
            logger.info(f"Using experiment: {experiment_name}")
            
            # Prepare training script and config
            train_script_path = self.prepare_training_script()
            azure_config_path = self.prepare_azure_config()
            
            if not train_script_path or not azure_config_path:
                logger.error("Failed to prepare training files")
                return False
            
            # Create run configuration
            run_config = RunConfiguration()
            run_config.environment = self.environment
            
            # Prepare source directory (entire project)
            project_dir = Path(__file__).resolve().parent.parent
            
            # Create script run configuration
            src = ScriptRunConfig(
                source_directory=str(project_dir),
                script=str(train_script_path.relative_to(project_dir)),
                arguments=['--config', str(azure_config_path.relative_to(project_dir))],
                compute_target=self.compute_target,
                environment=self.environment,
                run_config=run_config
            )
            
            # Submit experiment
            logger.info("Submitting training job to Azure ML...")
            run = self.experiment.submit(src)
            
            # Display submission details
            logger.info(f"Job submitted with run ID: {run.id}")
            logger.info(f"Run details URL: {run.get_portal_url()}")
            
            # Monitor run
            logger.info("Monitoring run progress...")
            try:
                # Use RunDetails widget if in Jupyter
                from IPython.display import display
                display(RunDetails(run).show())
            except (ImportError, NameError):
                # Otherwise, use text monitoring
                status = run.get_status()
                while status not in ['Completed', 'Failed', 'Canceled']:
                    print(f"Run status: {status}")
                    # Wait for 60 seconds before checking again
                    try:
                        run.wait_for_completion(show_output=True, timeout_seconds=60)
                        break
                    except:
                        status = run.get_status()
            
            # Get final status
            final_status = run.get_status()
            if final_status == 'Completed':
                logger.info("Training job completed successfully!")
                
                # Download results
                logger.info("Downloading results and checkpoints...")
                run.download_files(prefix='checkpoints/', output_directory='./azure_outputs/checkpoints')
                run.download_file('results.csv', output_file_path='./azure_outputs/results.csv')
                
                return True
            else:
                logger.error(f"Training job ended with status: {final_status}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            return False
    
    def clean_up_resources(self, delete_compute=False, delete_workspace=False):
        """Clean up Azure ML resources."""
        try:
            if delete_compute and self.compute_target:
                logger.info(f"Deleting compute target: {self.compute_target.name}...")
                self.compute_target.delete()
                logger.info("Compute target deleted")
            
            if delete_workspace and self.workspace:
                logger.warning(f"Deleting entire workspace: {self.workspace.name}...")
                self.workspace.delete(delete_dependent_resources=True, no_wait=False)
                logger.info("Workspace deleted")
                
            return True
        except Exception as e:
            logger.error(f"Failed to clean up resources: {e}")
            return False

def main():
    """Main function to run the deployer."""
    parser = argparse.ArgumentParser(description='Deploy DR-PETS project to Azure ML')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--clean', action='store_true', help='Clean up compute resources after training')
    parser.add_argument('--delete-workspace', action='store_true', help='Delete the entire workspace (DANGEROUS)')
    args = parser.parse_args()
    
    deployer = AzureMLDeployer(args.config)
    
    # Step 1: Connect to workspace
    if not deployer.connect_workspace():
        logger.error("Failed to connect to Azure ML workspace")
        return
    
    # Step 2: Set up compute target
    if not deployer.setup_compute_target():
        logger.error("Failed to set up compute target")
        return
    
    # Step 3: Set up environment
    if not deployer.setup_environment():
        logger.error("Failed to set up environment")
        return
    
    # Step 4: Submit training job
    if not deployer.submit_training_job():
        logger.error("Failed to submit training job")
        return
    
    # Step 5: Clean up resources if requested
    if args.clean or args.delete_workspace:
        deployer.clean_up_resources(
            delete_compute=args.clean,
            delete_workspace=args.delete_workspace
        )

if __name__ == "__main__":
    main()

