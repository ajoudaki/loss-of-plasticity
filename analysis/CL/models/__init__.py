"""
Model definitions for continual learning experiments.
"""

from .mlp import CustomizableMLP
from .cnn import ConfigurableCNN
from .resnet import ConfigurableResNet
from .vit import VisionTransformer

__all__ = ['CustomizableMLP', 'ConfigurableCNN', 'ConfigurableResNet', 'VisionTransformer']