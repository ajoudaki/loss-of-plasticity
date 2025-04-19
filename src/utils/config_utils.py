"""
Utilities for working with configurations.
"""
import os
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional

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
    if cfg.logging.use_wandb:
        # Prepare wandb config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            config=wandb_config
        )
        return True
    return False