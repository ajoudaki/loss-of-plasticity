"""
Custom layers and utilities for model construction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(activation_name):
    """
    Returns the activation function based on name
    
    Parameters:
        activation_name (str): Name of the activation function
        
    Returns:
        nn.Module: PyTorch activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'none': nn.Identity()
    }
    
    if activation_name.lower() not in activations:
        raise ValueError(f"Activation {activation_name} not supported. "
                         f"Choose from: {list(activations.keys())}")
    
    return activations[activation_name.lower()]

def get_normalization(norm_name, num_features):
    """
    Returns the normalization layer based on name
    
    Parameters:
        norm_name (str): Name of the normalization ('batch', 'layer', etc.)
        num_features (int): Number of features for the normalization layer
        
    Returns:
        nn.Module: PyTorch normalization module or None
    """
    if norm_name is None:
        return None
        
    normalizations = {
        'batch': nn.BatchNorm1d(num_features),
        'layer': nn.LayerNorm(num_features),
        'instance': nn.InstanceNorm1d(num_features),
        'group': nn.GroupNorm(min(32, num_features), num_features),
        'none': None
    }
    
    norm_key = str(norm_name).lower()
    if norm_key not in normalizations:
        raise ValueError(f"Normalization {norm_name} not supported. "
                         f"Choose from: {list(normalizations.keys())}")
    
    return normalizations[norm_key]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization from the paper:
    "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        # Scale
        return self.scale * x_normalized


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Transformer
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, E, H', W')
        # Rearrange to sequence of patches: [B, C, H, W] -> [B, N, C]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x