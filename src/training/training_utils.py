"""
Utility functions for neural network training.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from omegaconf import DictConfig

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
        # For MLP, the output layer is the last layer in the sequential model
        output_layer = model.model[-1]
    elif model_type == 'cnn':
        # For CNN, the output layer is typically the last fully connected layer
        output_layer = model.fc_layers[-1]
    elif model_type == 'resnet':
        # For ResNet, the output layer is the linear layer
        output_layer = model.fc
    elif model_type == 'vit':
        # For ViT, the output layer is the head
        output_layer = model.head
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Only reinitialize weights for task classes
    with torch.no_grad():
        # Reinitialize weights for task classes
        layer_norm = (output_layer.weight**2).mean().item()**0.5
        for cls in range(len(output_layer.weight)):
            # Initialize the weights for this class, use layer std
            nn.init.normal_(output_layer.weight[cls], std=layer_norm)
            # Initialize the bias for this class
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias[cls])
    
    print(f"Reinitialized output weights for classes: {task_classes}")

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
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")