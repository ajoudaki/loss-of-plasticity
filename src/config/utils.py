"""
Utilities for working with configurations and model setup.
"""
import os
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional, List

def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_str: Device string (e.g., 'cuda', 'cpu', 'mps')
               If None, will select the best available device
    
    Returns:
        torch.device: The selected device
    """
    if device_str is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)

def setup_wandb(cfg: DictConfig) -> bool:
    """
    Setup weights & biases logging.
    
    Args:
        cfg: Configuration object
        
    Returns:
        bool: True if wandb was initialized, False otherwise
    """
    if cfg.use_wandb:
        # Prepare wandb config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        
        # Create a descriptive run name with the requested parameters
        model_name = cfg.model.name
        dataset_name = cfg.dataset.name
        training_type = cfg.training.training_type
        
        # Initialize wandb with optional entity parameter and the created run name
        init_args = {
            "project": cfg.wandb_project,
            "tags": cfg.wandb_tags + [training_type],
            "config": wandb_config,
        }
        
        # Add entity parameter if it exists
        if hasattr(cfg.logging, "wandb_entity"):
            init_args["entity"] = cfg.logging.wandb_entity
            
        wandb.init(**init_args)
        return True
    return False

def create_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.
    
    Args:
        model: The model whose parameters will be optimized
        cfg: Configuration object containing optimizer settings
        
    Returns:
        Optimizer instance
    """
    optimizer_name = cfg.optimizer.name.lower()
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            dampening=cfg.optimizer.dampening,
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=cfg.optimizer.nesterov
        )
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.optimizer.lr,
            alpha=cfg.optimizer.alpha,
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=cfg.optimizer.momentum,
            centered=cfg.optimizer.centered
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")