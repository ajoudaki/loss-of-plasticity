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
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    
    model_params['num_classes'] = cfg.dataset.num_classes
    
    if model_name == 'mlp' or model_name == 'gated_mlp':
        model_params['input_size'] = cfg.dataset.input_size
        model_params['output_size'] = cfg.dataset.num_classes
        model_params['eigenval_reg_momentum'] = cfg.training.get('eigenval_reg_momentum', 0.9)
        model_params['eigenval_reg_lambda'] = cfg.training.get('eigenval_reg_lambda', 0.1)
        return MLP(**model_params)
    
    elif model_name == 'cnn':
        model_params['in_channels'] = cfg.dataset.in_channels
        model_params['input_size'] = cfg.dataset.img_size
        return CNN(**model_params)
    
    elif model_name == 'resnet':
        model_params['in_channels'] = cfg.dataset.in_channels
        return ResNet(**model_params)
    
    elif model_name == 'vit':
        model_params['img_size'] = cfg.dataset.img_size
        model_params['in_channels'] = cfg.dataset.in_channels
        return VisionTransformer(**model_params)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
