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
    
    # Convert model config to dict and filter relevant parameters
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Remove name and _target_ from parameters
    if 'name' in model_params:
        del model_params['name']
    if '_target_' in model_params:
        del model_params['_target_']
    
    # Add dataset parameters
    # These parameters are always set regardless of model type
    model_params['num_classes'] = cfg.dataset.num_classes
    
    # Create the appropriate model type with filtered parameters
    if model_name == 'mlp':
        # MLP specific parameters
        model_params['input_size'] = cfg.dataset.input_size
        model_params['output_size'] = cfg.dataset.num_classes
        
        # Filter parameters to only include those used by MLP
        mlp_params = {
            'hidden_sizes': model_params['hidden_sizes'],
            'activation': model_params['activation'],
            'dropout_p': model_params['dropout_p'],
            'normalization': model_params['normalization'],
            'norm_after_activation': model_params['norm_after_activation'],
            'bias': model_params['bias'],
            'normalization_affine': model_params['normalization_affine'],
            'input_size': model_params['input_size'],
            'output_size': model_params['output_size']
        }
        return MLP(**mlp_params)
    
    elif model_name == 'cnn':
        # CNN specific parameters
        model_params['in_channels'] = cfg.dataset.in_channels
        model_params['input_size'] = cfg.dataset.img_size
        
        # Filter parameters to only include those used by CNN
        cnn_params = {
            'conv_channels': model_params['conv_channels'],
            'kernel_sizes': model_params['kernel_sizes'],
            'strides': model_params['strides'],
            'paddings': model_params['paddings'],
            'fc_hidden_units': model_params['fc_hidden_units'],
            'activation': model_params['activation'],
            'dropout_p': model_params['dropout_p'],
            'pool_type': model_params['pool_type'],
            'pool_size': model_params['pool_size'],
            'use_batchnorm': model_params['use_batchnorm'],
            'norm_after_activation': model_params['norm_after_activation'],
            'normalization_affine': model_params['normalization_affine'],
            'in_channels': model_params['in_channels'],
            'num_classes': model_params['num_classes'],
            'input_size': model_params['input_size']
        }
        return CNN(**cnn_params)
    
    elif model_name == 'resnet':
        # ResNet specific parameters
        model_params['in_channels'] = cfg.dataset.in_channels
        
        # Filter parameters to only include those used by ResNet
        resnet_params = {
            'layers': model_params['layers'],
            'base_channels': model_params['base_channels'],
            'activation': model_params['activation'],
            'dropout_p': model_params['dropout_p'],
            'use_batchnorm': model_params['use_batchnorm'],
            'norm_after_activation': model_params['norm_after_activation'],
            'normalization_affine': model_params['normalization_affine'],
            'in_channels': model_params['in_channels'],
            'num_classes': model_params['num_classes']
        }
        return ResNet(**resnet_params)
    
    elif model_name == 'vit':
        # ViT specific parameters
        model_params['img_size'] = cfg.dataset.img_size
        model_params['in_channels'] = cfg.dataset.in_channels
        
        # Filter parameters to only include those used by ViT
        vit_params = {
            'patch_size': model_params['patch_size'],
            'embed_dim': model_params['embed_dim'],
            'depth': model_params['depth'],
            'n_heads': model_params['n_heads'],
            'mlp_ratio': model_params['mlp_ratio'],
            'qkv_bias': model_params['qkv_bias'],
            'drop_rate': model_params['drop_rate'],
            'attn_drop_rate': model_params['attn_drop_rate'],
            'activation': model_params['activation'],
            'normalization': model_params['normalization'],
            'normalization_affine': model_params['normalization_affine'],
            'img_size': model_params['img_size'],
            'in_channels': model_params['in_channels'],
            'num_classes': model_params['num_classes']
        }
        return VisionTransformer(**vit_params)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")