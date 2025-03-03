"""
Utility functions for continual learning experiments.
"""

from .config import ExperimentConfig, load_config, save_config
from .visualization import (
    plot_learning_curves,
    plot_task_performance,
    plot_forgetting_curves,
    plot_activation_heatmap
)

__all__ = [
    'ExperimentConfig',
    'load_config',
    'save_config',
    'plot_learning_curves',
    'plot_task_performance',
    'plot_forgetting_curves',
    'plot_activation_heatmap'
]