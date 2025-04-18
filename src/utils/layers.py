import torch
import torch.nn as nn
import random
import numpy as np

def get_activation(activation_name):
    """Returns the activation function based on name."""
    activations = {
        'relu': nn.ReLU(inplace=False),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=False),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(inplace=False),
        'selu': nn.SELU(inplace=False),
        'none': nn.Identity()
    }
    
    if activation_name.lower() not in activations:
        raise ValueError(f"Activation {activation_name} not supported. "
                     f"Choose from: {list(activations.keys())}")
    
    return activations[activation_name.lower()]

def get_normalization(norm_name, num_features, affine=True):
    """Returns the normalization layer based on name."""
    if norm_name is None:
        return None
        
    normalizations = {
        'batch': nn.BatchNorm1d(num_features, affine=affine),
        'batch2d': nn.BatchNorm2d(num_features, affine=affine),
        'layer': nn.LayerNorm(num_features, elementwise_affine=affine),
        'instance': nn.InstanceNorm1d(num_features, affine=affine),
        'instance2d': nn.InstanceNorm2d(num_features, affine=affine),
        'group': nn.GroupNorm(min(32, num_features), num_features, affine=affine),
        'none': nn.Identity()
    }
    
    norm_key = str(norm_name).lower()
    if norm_key not in normalizations:
        raise ValueError(f"Normalization {norm_name} not supported. "
                     f"Choose from: {list(normalizations.keys())}")
    
    return normalizations[norm_key]

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
