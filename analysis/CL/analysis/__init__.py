"""
Analysis tools for neural network interpretability in continual learning.
"""

from .activation_analysis import (
    analyze_activations,
    compute_activation_statistics, 
    visualize_activation_distributions
)
from .rank_analysis import (
    analyze_rank_dynamics,
    compute_layer_ranks,
    compute_feature_diversity
)

__all__ = [
    'analyze_activations',
    'compute_activation_statistics',
    'visualize_activation_distributions',
    'analyze_rank_dynamics',
    'compute_layer_ranks',
    'compute_feature_diversity'
]