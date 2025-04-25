"""
Registers all structured configurations with Hydra's ConfigStore.
This enables proper validation and type checking of configuration values.
"""

from hydra.core.config_store import ConfigStore
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
    LoggingConfig
)

def register_configs():
    """Register all configuration schemas with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    
    # Register the main experiment config schema
    cs.store(name="experiment_schema", node=ExperimentConfig)
    
    # Register component schemas
    cs.store(group="model", name="mlp_schema", node=MLPConfig)
    cs.store(group="model", name="cnn_schema", node=CNNConfig)
    cs.store(group="model", name="resnet_schema", node=ResNetConfig)
    cs.store(group="model", name="vit_schema", node=ViTConfig)
    
    cs.store(group="dataset", name="dataset_schema", node=DatasetConfig)
    cs.store(group="optimizer", name="optimizer_schema", node=OptimizerConfig)
    cs.store(group="metrics", name="metrics_schema", node=MetricsConfig)
    cs.store(group="training", name="training_schema", node=TrainingConfig)
    cs.store(group="logging", name="logging_schema", node=LoggingConfig)
    
    return cs