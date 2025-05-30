"""
Config schemas for Hydra structured configs.
These dataclasses define the structure and defaults for all configuration options.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import MISSING


@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron model."""
    _target_: str = "src.models.MLP"
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    dropout_p: float = 0.1
    normalization: str = "batch"
    norm_after_activation: bool = False
    bias: bool = True
    normalization_affine: bool = True
    input_size: Optional[int] = None  # Will be set based on dataset
    output_size: Optional[int] = None  # Will be set based on dataset


@dataclass
class CNNConfig:
    """Configuration for Convolutional Neural Network model."""
    _target_: str = "src.models.CNN"
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [1, 1, 1])
    fc_hidden_units: List[int] = field(default_factory=lambda: [512])
    activation: str = "relu"
    dropout_p: float = 0.1
    pool_type: str = "max"
    pool_size: int = 2
    normalization: str = "batch"
    norm_after_activation: bool = False
    normalization_affine: bool = True
    input_size: Optional[int] = None  # Will be set based on dataset
    in_channels: Optional[int] = None  # Will be set based on dataset
    num_classes: Optional[int] = None  # Will be set based on dataset


@dataclass
class ResNetConfig:
    """Configuration for ResNet model."""
    _target_: str = "src.models.ResNet"
    layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    base_channels: int = 64
    activation: str = "relu"
    dropout_p: float = 0.1
    normalization: str = "batch"
    norm_after_activation: bool = False
    normalization_affine: bool = True
    in_channels: Optional[int] = None  # Will be set based on dataset
    num_classes: Optional[int] = None  # Will be set based on dataset


@dataclass
class ViTConfig:
    """Configuration for Vision Transformer model."""
    _target_: str = "src.models.VisionTransformer"
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout_p: float = 0.1
    attn_drop_rate: float = 0.1
    activation: str = "gelu"
    normalization: str = "layer"
    normalization_affine: bool = True
    img_size: Optional[int] = None  # Will be set based on dataset
    in_channels: Optional[int] = None  # Will be set based on dataset
    num_classes: Optional[int] = None  # Will be set based on dataset


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    name: str = "mlp"
    # Common parameters
    _target_: str = "src.models.MLP"  # Will be set based on model name
    
    # Parameters for all model types
    # MLP parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # General parameters that exist across models
    activation: str = "relu"
    dropout_p: float = 0.1
    normalization: str = "batch"
    norm_after_activation: bool = False
    normalization_affine: bool = True
    bias: bool = True
    
    # CNN parameters
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 1, 1])
    paddings: List[int] = field(default_factory=lambda: [1, 1, 1])
    fc_hidden_units: List[int] = field(default_factory=lambda: [512])
    pool_type: str = "max"
    pool_size: int = 2
    
    # ResNet parameters
    layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    base_channels: int = 64
    
    # ViT parameters
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    attn_drop_rate: float = 0.1
    
    # Dynamic parameters that will be set based on dataset
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    in_channels: Optional[int] = None
    num_classes: Optional[int] = None
    img_size: Optional[int] = None


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    name: str = "cifar10"
    input_size: int = MISSING  # Will be set based on dataset
    img_size: int = MISSING    # Will be set based on dataset
    in_channels: int = MISSING  # Will be set based on dataset 
    num_classes: int = MISSING  # Will be set based on dataset


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    
    # Adam parameters
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    
    # SGD parameters  
    momentum: float = 0.9
    dampening: float = 0.0
    nesterov: bool = False
    
    # RMSprop parameters
    alpha: float = 0.99
    centered: bool = False
    
    # Special flags
    reinit_optimizer: bool = False  # Reinitialize optimizer state for each new task


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and thresholds."""
    metrics_frequency: int = 5
    dead_threshold: float = 0.95
    corr_threshold: float = 0.95
    saturation_threshold: float = 1e-4
    saturation_percentage: float = 0.99
    gaussianity_method: str = "shapiro"  # Options: "shapiro", "ks", "anderson", "kurtosis"
    log_activation_histograms: bool = True
    monitor_filters: List[str] = field(default_factory=list)  # Layer name patterns to monitor


@dataclass
class TrainingConfig:
    """Configuration for training procedures."""
    training_type: str = "standard"  # Options: "finetune", "replay", "ewc", "lwf"
    epochs_per_task: int = 20
    batch_size: int = 128
    num_workers: int = 2  # Number of workers for data loading
    no_augment: bool = False
    early_stopping_steps: int = 0
    reset: bool = False  # Reset model weights before training on each new task
    seed: int = 42
    device: Optional[str] = None  # 'cuda', 'cpu', or 'mps'
    # Task configuration (formerly in TaskConfig)
    tasks: int = 10  # Number of tasks
    classes_per_task: int = 2  # Classes per task
    partitions: Optional[List[List[int]]] = None  # Custom class partitions


@dataclass
class LoggingConfig:
    """Configuration for experiment logging."""
    wandb_entity: Optional[str] = None
    summary: bool = True  # Show summary after each task


@dataclass
class ExperimentConfig:
    """Master configuration for experiments."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dryrun: bool = False
    use_wandb: bool = False
    wandb_project: str = "continual-learning-experiment" 
    wandb_tags: List[str] = field(default_factory=list)