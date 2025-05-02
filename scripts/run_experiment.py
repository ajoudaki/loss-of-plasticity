#!/usr/bin/env python3
"""
Script to run experiments with the Neural Network Dynamic Scaling codebase.
This script uses Hydra for configuration management.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional
import wandb

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
from src.models.layers import set_seed
from src.training import train_continual_learning, train_cloning_experiment
from src.config import register_configs
from src.utils.data import prepare_continual_learning_dataloaders
from src.config.utils import get_device, setup_wandb
from src.models.model_factory import create_model

# Register all configurations with Hydra
register_configs()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run_experiment(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    """
    Run a neural network experiment with the specified configuration.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Optional[Dict[str, Any]]: Training history if successful, None for dry run
    """
    
    # Initialize W&B if requested and available
    use_wandb = setup_wandb(cfg)
    
    # Print configuration
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Prepare dataloaders for continual learning
    task_dataloaders, num_classes, class_sequence = prepare_continual_learning_dataloaders(cfg)
    
    # Get device
    device = get_device(cfg.training.device)
    # Store as string representation instead of device object
    cfg.training.device = str(device)
    print(f"Using device: {device}")
    
    # Create model and move it to the device
    model = create_model(cfg).to(device)
    print(f"Created {cfg.model.name.upper()} model")
    
    
    # Check for dryrun
    if cfg.dryrun:
        print("Dry run completed, exiting without training.")
        return None
    
    # Train using the appropriate training method based on config
    if hasattr(cfg.training, 'expansion_factor') and hasattr(cfg.training, 'num_expansions'):
        # Use cloning experiment training
        history = train_cloning_experiment(model, task_dataloaders, cfg, device=device)
    else:
        # Use standard continual learning training
        history = train_continual_learning(model, task_dataloaders, cfg, device=device)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 
              f'models/{cfg.model.name}_{cfg.dataset.name}_{cfg.training.tasks}tasks.pth')
    
    # Finish W&B run
    if use_wandb:
        wandb.finish()
    
    return history

if __name__ == "__main__":
    run_experiment()