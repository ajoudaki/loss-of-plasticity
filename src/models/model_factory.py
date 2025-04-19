"""
Factory module for creating neural network models.
"""
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional

from . import MLP, CNN, ResNet, VisionTransformer

def create_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function to create models based on configuration.
    
    Args:
        cfg: Configuration object containing model and dataset specifications
        
    Returns:
        An initialized PyTorch model
    """
    model_name = cfg.model.name.lower()
    model_params = {}
    
    # Get model config based on name and convert to dict
    if hasattr(cfg.model, model_name):
        model_config = getattr(cfg.model, model_name)
        model_params = OmegaConf.to_container(model_config, resolve=True)
        # Remove _target_ if present
        if '_target_' in model_params:
            del model_params['_target_']
    
    # Add dataset parameters based on model type
    if model_name == 'mlp':
        model_params['input_size'] = cfg.dataset.input_size
        model_params['output_size'] = cfg.dataset.num_classes
        return MLP(**model_params)
    
    elif model_name == 'cnn':
        model_params['in_channels'] = cfg.dataset.in_channels
        model_params['num_classes'] = cfg.dataset.num_classes
        model_params['input_size'] = cfg.dataset.img_size
        return CNN(**model_params)
    
    elif model_name == 'resnet':
        model_params['num_classes'] = cfg.dataset.num_classes
        model_params['in_channels'] = cfg.dataset.in_channels
        return ResNet(**model_params)
    
    elif model_name == 'vit':
        model_params['img_size'] = cfg.dataset.img_size
        model_params['in_channels'] = cfg.dataset.in_channels
        model_params['num_classes'] = cfg.dataset.num_classes
        return VisionTransformer(**model_params)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")