# Azure ML Deployment for DR-PETS

This guide provides instructions for deploying the DR-PETS project on Azure Machine Learning. The deployment scripts automate the process of creating an Azure ML workspace, configuring GPU-enabled compute resources, and running training jobs with proper experiment tracking.

## Prerequisites

Before using the Azure ML deployment scripts, ensure you have:

1. **Azure Account**: An active Azure subscription with sufficient quota for GPU VMs
2. **Azure CLI**: Installed and configured with `az login`
3. **Python Dependencies**: Install the required Python packages:
   ```bash
   pip install azureml-core azureml-mlflow azureml-widgets pyyaml
   ```

## Setup and Authentication

### Azure ML Authentication

The deployment script supports several authentication methods:

1. **Environment Variables** (Recommended):
   ```bash
   export AZURE_SUBSCRIPTION_ID="your-subscription-id"
   export AZURE_RESOURCE_GROUP="your-resource-group"
   ```

2. **Configuration File**:
   Edit `deploy/azure_config.yaml` and add your subscription ID and resource group:
   ```yaml
   azure:
     subscription_id: "your-subscription-id"
     resource_group: "your-resource-group"
   ```

3. **Azure CLI Authentication**:
   Make sure you're logged in with Azure CLI:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

### First-Time Setup

If you're using Azure ML for the first time, you may need to register the Machine Learning service provider:

```bash
az provider register --namespace Microsoft.MachineLearningServices
```

## Usage Examples

### Basic Deployment

To deploy DR-PETS with default settings:

```bash
python deploy/azure_deploy.py
```

This will:
1. Create or connect to an Azure ML workspace
2. Set up a GPU compute cluster (Standard_NC6 by default)
3. Configure the training environment with all dependencies
4. Submit the training job and monitor progress
5. Download results when training completes

### Custom Configuration

To use a custom configuration:

```bash
python deploy/azure_deploy.py --config deploy/my_custom_azure_config.yaml
```

### Cleaning Up Resources

To clean up compute resources after training (to avoid ongoing charges):

```bash
python deploy/azure_deploy.py --clean
```

**Warning**: To delete the entire workspace (use with caution):

```bash
python deploy/azure_deploy.py --delete-workspace
```

## Cost Management

### VM Pricing Information

The deployment uses auto-scaling compute clusters that scale down to zero nodes when not in use. Default configuration uses:

| VM Size | GPU | vCPUs | Memory |  
|---------|-----|-------|--------|
| Standard_NC6 | 1x K80 | 6 | 56 GB |  
| Standard_NC12 | 2x K80 | 12 | 112 GB |  
| Standard_NC24 | 4x K80 | 24 | 224 GB |  
| Standard_ND40rs_v2 | 8x V100 | 40 | 672 GB |  

**Note**: Prices are approximate and may vary by region. Check the [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/) for current rates.

### Budget Control

The deployment script includes budget control mechanisms:

1. **Auto-scaling**: Compute scales to 0 nodes when idle (default: after 20 minutes)
2. **Budget Limit**: Set maximum budget in `azure_config.yaml`:
   ```yaml
   azure:
     budget_limit_usd: 50.0
   ```
3. **Auto-termination**: Enable automatic termination when training completes:
   ```yaml
   azure:
     auto_terminate: true
   ```

## Monitoring Training Progress

The Azure ML deployment provides several ways to monitor training:

### Azure ML Studio

Monitor experiments through the Azure ML Studio web interface:
1. Go to [Azure ML Studio](https://ml.azure.com)
2. Select your workspace
3. Navigate to "Experiments" to see your training run

### TensorBoard

The deployment automatically configures TensorBoard logging:

1. In Azure ML Studio, go to your run
2. Click the "TensorBoard" tab to view metrics

### Logs and Metrics

- **Metrics**: View metrics in the Azure ML Studio "Metrics" tab
- **Logs**: Check logs in the "Outputs + logs" tab
- **Artifacts**: Access saved models in the "Outputs + logs" tab under "Outputs"

## Hyperparameter Tuning

The configuration supports hyperparameter tuning:

1. Enable tuning in `azure_config.yaml`:
   ```yaml
   experiment:
     hyperparameter_tuning:
       enabled: true
       parameters:
         epsilon:
           distribution: uniform
           min_value: 0.01
           max_value: 0.1
   ```

2. Run the deployment script with the `--tune` flag:
   ```bash
   python deploy/azure_deploy.py --tune
   ```

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Ensure you've logged in with `az login`
   - Check that your subscription ID and resource group are correct
   - Verify you have sufficient permissions (Contributor role)

2. **Quota Limits**:
   - If you receive errors about quota, request a quota increase for GPU VMs in your region
   - Try a different region or a different VM size

3. **Environment Setup Failures**:
   - Check the run logs in Azure ML Studio for package installation errors
   - Verify your dependencies are compatible with the selected Python version

4. **Connection Timeouts**:
   - Azure ML operations can take time; increase timeout settings
   - Check your network connection

### Getting Support

1. Check Azure ML documentation: https://docs.microsoft.com/en-us/azure/machine-learning/
2. Azure ML SDK reference: https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py
3. File issues on the DR-PETS GitHub repository

## Advanced Configuration

### Custom Docker Images

For advanced users, you can specify a custom Docker base image:

```yaml
environment:
  custom_docker_image: "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04"
```

### Low-Priority VMs

To reduce costs, you can use low-priority VMs (subject to preemption):

```yaml
azure:
  vm_priority: lowpriority
```

### Distributed Training

For distributed training across multiple nodes:

```yaml
experiment:
  distributed:
    enabled: true
    framework: pytorch
    process_count_per_node: 1
```

## References

- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)
- [Azure VM Sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-gpu)

