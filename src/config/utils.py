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
    if cfg.logging.use_wandb:
        # Prepare wandb config
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        
        # Create a descriptive run name with the requested parameters
        model_name = cfg.model.name
        
        
        dropout = cfg.model.dropout_p if model_name != 'vit' else cfg.model.drop_rate
        
        # Get depth based on model type
        if model_name == 'mlp':
            depth = len(cfg.model.hidden_sizes)
        elif model_name == 'cnn':
            depth = len(cfg.model.conv_channels)
        elif model_name == 'resnet':
            depth = len(cfg.model.layers)
        elif model_name == 'vit':
            depth = cfg.model.depth
        else:
            depth = 0
        
        # Override normalization for CNN and ResNet (they use use_batchnorm flag)
        if model_name in ['cnn', 'resnet']:
            normalization = "batchnorm" if cfg.model.use_batchnorm else "none"
        else:
            # Get normalization type based on model
            normalization = cfg.model.normalization
        
        # Determine if we're resetting all weights or just output weights
        reset_type = "all_reset" if cfg.training.reset else ("output_reset" if cfg.training.reinit_output else "no_reset")
        
        # Add timestamp to ensure unique run names
        import time
        timestamp = int(time.time())
        
        # Create run name with all requested parameters and timestamp
        run_name = f"{model_name}_{normalization}_drop{dropout}_depth{depth}_{reset_type}_cls{cfg.training.classes_per_task}_{timestamp}"
        
        # Initialize wandb with optional entity parameter and the created run name
        init_args = {
            "project": cfg.logging.wandb_project,
            "config": wandb_config,
            "name": run_name
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

def reinitialize_output_weights(model: nn.Module, task_classes: List[int], model_type: str = 'mlp') -> None:
    """
    Reinitialize the output weights for the specified task classes.
    
    Args:
        model: The neural network model
        task_classes: List of class indices for the current task
        model_type: Type of model ('mlp', 'cnn', 'resnet', or 'vit')
    """
    # Get the output layer
    if model_type == 'mlp':
        # For MLP, the output layer is accessible through the layers ModuleDict with key 'out'
        output_layer = model.layers['out']
    elif model_type == 'cnn':
        # For CNN, the output layer is the final fc layer in the ModuleDict
        output_layer = model.layers['fc_out']
    elif model_type == 'resnet':
        # For ResNet, the output layer is the linear layer in the layers ModuleDict
        output_layer = model.layers['fc']
    elif model_type == 'vit':
        # For ViT, the output layer is the head in the ModuleDict
        output_layer = model.layers['head']
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Only reinitialize weights for task classes
    with torch.no_grad():
        # Calculate the current layer norm for proper scaling
        layer_norm = (output_layer.weight**2).mean().item()**0.5
        
        # Reinitialize weights only for the specific task classes
        for cls in task_classes:
            if cls < len(output_layer.weight):  # Ensure class index is valid
                # Initialize the weights for this class, use layer std
                nn.init.normal_(output_layer.weight[cls], std=layer_norm)
                # Initialize the bias for this class
                if output_layer.bias is not None:
                    nn.init.zeros_(output_layer.bias[cls])
    
    print(f"Reinitialized output weights for classes: {task_classes}")