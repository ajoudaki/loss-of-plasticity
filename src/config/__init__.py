"""
Configuration module for the NN-dynamic-scaling project.
Provides schema definitions, registration, and utilities for configuration management.
"""

from .schema import (
    ExperimentConfig,
    ModelConfig,
    MLPConfig,
    CNNConfig,
    ResNetConfig,
    ViTConfig,
    DatasetConfig,
    OptimizerConfig,
    MetricsConfig,
    TrainingConfig,
    TaskConfig,
    LoggingConfig
)

from .registry import register_configs
from .utils import (
    get_device,
    setup_wandb,
    create_optimizer,
    reinitialize_output_weights
)

__all__ = [
    # Schema classes
    'ExperimentConfig',
    'ModelConfig',
    'MLPConfig',
    'CNNConfig',
    'ResNetConfig',
    'ViTConfig',
    'DatasetConfig',
    'OptimizerConfig',
    'MetricsConfig',
    'TrainingConfig',
    'TaskConfig',
    'LoggingConfig',
    
    # Registration function
    'register_configs',
    
    # Utility functions
    'get_device',
    'setup_wandb',
    'create_optimizer',
    'reinitialize_output_weights'
]