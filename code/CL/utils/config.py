"""
Configuration utilities for continual learning experiments.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union


@dataclass
class ExperimentConfig:
    """Configuration for continual learning experiments."""
    
    # General experiment settings
    experiment_name: str = "default_experiment"
    random_seed: int = 42
    use_cuda: bool = True
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    use_wandb: bool = False
    wandb_project: str = "continual_learning"
    wandb_entity: Optional[str] = None
    
    # Dataset settings
    dataset_name: str = "cifar10"
    dataset_root: str = "./data"
    input_size: int = 32
    num_classes: int = 10
    
    # Task settings
    task_type: str = "pairs"  # 'pairs' or 'sequential'
    class_sequence: Optional[List[int]] = None
    num_tasks: Optional[int] = None
    
    # Model settings
    model_type: str = "mlp"  # 'mlp', 'cnn', 'resnet', 'vit'
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_sizes": [512, 256, 128],
        "activation": "relu",
        "dropout_p": 0.0,
        "normalization": None,
        "norm_after_activation": False
    })
    
    # Training settings
    optimizer_type: str = "adam"  # 'sgd', 'adam', 'adamw'
    optimizer_config: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 0.001,
        "weight_decay": 0.0
    })
    scheduler_type: Optional[str] = None  # 'step', 'cosine', 'plateau', None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 128
    num_workers: int = 2
    num_epochs_per_task: int = 10
    early_stopping: Optional[int] = None
    
    # Metrics and analysis settings
    metrics_config: Dict[str, Any] = field(default_factory=lambda: {
        "compute_rank_metrics": True,
        "compute_component_metrics": True,
        "compute_activation_stats": True,
        "compute_weight_stats": True,
        "track_gradient_stats": False,
        "record_activations": True,
        "record_activations_freq": 5,  # epochs between recording
        "record_sample_size": 100,  # number of samples for recording
        "compute_forgetting": True,  # track performance on previous tasks
    })
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)


def load_config(path):
    """
    Load experiment configuration from file.
    
    Parameters:
        path (str): Path to configuration file (yaml or json)
        
    Returns:
        ExperimentConfig: Loaded configuration
    """
    _, ext = os.path.splitext(path)
    
    with open(path, 'r') as f:
        if ext.lower() == '.yaml' or ext.lower() == '.yml':
            config_dict = yaml.safe_load(f)
        elif ext.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    return ExperimentConfig.from_dict(config_dict)


def save_config(config, path):
    """
    Save experiment configuration to file.
    
    Parameters:
        config (ExperimentConfig): Configuration to save
        path (str): Path to save configuration file (yaml or json)
    """
    config_dict = config.to_dict()
    _, ext = os.path.splitext(path)
    
    with open(path, 'w') as f:
        if ext.lower() == '.yaml' or ext.lower() == '.yml':
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif ext.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


def create_model_from_config(config):
    """
    Create model based on configuration.
    
    Parameters:
        config (ExperimentConfig): Experiment configuration
        
    Returns:
        nn.Module: Initialized model
    """
    from models import CustomizableMLP, ConfigurableCNN, ConfigurableResNet, VisionTransformer
    
    if config.model_type.lower() == 'mlp':
        # Get input size based on dataset
        if config.dataset_name.lower() in ['mnist', 'fashion_mnist']:
            input_size = 28 * 28
        elif config.dataset_name.lower() in ['cifar10', 'cifar100']:
            input_size = 32 * 32 * 3
        else:
            input_size = config.input_size * config.input_size * 3
        
        model_config = {
            'input_size': input_size,
            'hidden_sizes': config.model_config.get('hidden_sizes', [512, 256, 128]),
            'output_size': config.num_classes,
            'activation': config.model_config.get('activation', 'relu'),
            'dropout_p': config.model_config.get('dropout_p', 0.0),
            'normalization': config.model_config.get('normalization', None),
            'norm_after_activation': config.model_config.get('norm_after_activation', False),
            'record_activations': config.metrics_config.get('record_activations', False),
        }
        
        return CustomizableMLP(**model_config)
    
    elif config.model_type.lower() == 'cnn':
        # Get input channels based on dataset
        if config.dataset_name.lower() in ['mnist', 'fashion_mnist']:
            in_channels = 1
        else:
            in_channels = 3
        
        model_config = {
            'in_channels': in_channels,
            'conv_channels': config.model_config.get('conv_channels', [64, 128, 256]),
            'kernel_sizes': config.model_config.get('kernel_sizes', [3, 3, 3]),
            'fc_hidden_units': config.model_config.get('fc_hidden_units', [512]),
            'num_classes': config.num_classes,
            'input_size': config.input_size,
            'activation': config.model_config.get('activation', 'relu'),
            'dropout_p': config.model_config.get('dropout_p', 0.0),
            'use_batchnorm': config.model_config.get('normalization', None) == 'batch',
            'norm_after_activation': config.model_config.get('norm_after_activation', False),
            'record_activations': config.metrics_config.get('record_activations', False),
        }
        
        return ConfigurableCNN(**model_config)
    
    elif config.model_type.lower() == 'resnet':
        # Get input channels based on dataset
        if config.dataset_name.lower() in ['mnist', 'fashion_mnist']:
            in_channels = 1
        else:
            in_channels = 3
        
        model_config = {
            'in_channels': in_channels,
            'num_classes': config.num_classes,
            'activation': config.model_config.get('activation', 'relu'),
            'dropout_p': config.model_config.get('dropout_p', 0.0),
            'use_batchnorm': config.model_config.get('normalization', None) == 'batch',
            'norm_after_activation': config.model_config.get('norm_after_activation', False),
            'record_activations': config.metrics_config.get('record_activations', False),
        }
        
        # Optional ResNet-specific parameters
        if 'layers' in config.model_config:
            model_config['layers'] = config.model_config['layers']
        if 'base_channels' in config.model_config:
            model_config['base_channels'] = config.model_config['base_channels']
        
        return ConfigurableResNet(**model_config)
    
    elif config.model_type.lower() == 'vit':
        # Get input channels based on dataset
        if config.dataset_name.lower() in ['mnist', 'fashion_mnist']:
            in_channels = 1
        else:
            in_channels = 3
        
        model_config = {
            'img_size': config.input_size,
            'patch_size': config.model_config.get('patch_size', 4),
            'in_channels': in_channels,
            'num_classes': config.num_classes,
            'embed_dim': config.model_config.get('embed_dim', 192),
            'depth': config.model_config.get('depth', 12),
            'n_heads': config.model_config.get('n_heads', 8),
            'mlp_ratio': config.model_config.get('mlp_ratio', 4.0),
            'qkv_bias': config.model_config.get('qkv_bias', True),
            'drop_rate': config.model_config.get('dropout_p', 0.0),
            'activation': config.model_config.get('activation', 'gelu'),
            'normalization': config.model_config.get('normalization', 'layer'),
            'record_activations': config.metrics_config.get('record_activations', False),
        }
        
        return VisionTransformer(**model_config)
    
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")


def create_optimizer_from_config(config, model):
    """
    Create optimizer based on configuration.
    
    Parameters:
        config (ExperimentConfig): Experiment configuration
        model (nn.Module): Model to optimize
        
    Returns:
        torch.optim.Optimizer: Initialized optimizer
    """
    import torch.optim as optim
    
    lr = config.optimizer_config.get('lr', 0.001)
    weight_decay = config.optimizer_config.get('weight_decay', 0.0)
    
    if config.optimizer_type.lower() == 'sgd':
        momentum = config.optimizer_config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif config.optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif config.optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")


def create_scheduler_from_config(config, optimizer):
    """
    Create learning rate scheduler based on configuration.
    
    Parameters:
        config (ExperimentConfig): Experiment configuration
        optimizer (torch.optim.Optimizer): Optimizer
        
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Initialized scheduler or None
    """
    import torch.optim as optim
    
    if config.scheduler_type is None:
        return None
    
    elif config.scheduler_type.lower() == 'step':
        step_size = config.scheduler_config.get('step_size', 30)
        gamma = config.scheduler_config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif config.scheduler_type.lower() == 'cosine':
        T_max = config.scheduler_config.get('T_max', config.num_epochs_per_task)
        eta_min = config.scheduler_config.get('eta_min', 0)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif config.scheduler_type.lower() == 'plateau':
        patience = config.scheduler_config.get('patience', 5)
        factor = config.scheduler_config.get('factor', 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")