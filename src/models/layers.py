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

class TransformerBatchNorm(nn.Module):
    """
    BatchNorm designed specifically for Transformer architectures.
    Handles inputs of shape (batch_size, sequence_length, hidden_dim).
    """
    def __init__(self, hidden_dim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        # Use standard BatchNorm1d under the hood
        self.bn = nn.BatchNorm1d(hidden_dim, eps=eps, momentum=momentum, 
                                affine=affine, track_running_stats=track_running_stats)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, hidden_dim)
        # BatchNorm1d expects: (batch_size, hidden_dim, seq_len)
        
        # Remember original shape
        orig_shape = x.shape
        
        # Transpose to get hidden_dim in the correct position for BatchNorm1d
        x = x.permute(0, 2, 1)
        
        # Apply batch normalization
        x = self.bn(x)
        
        # Transpose back to original shape
        x = x.permute(0, 2, 1)
        
        return x
    
def get_normalization(norm_name, num_features, affine=True,model=None):
    """Returns the normalization layer based on name."""
    if norm_name is None:
        return None
    if model=='vit':
        if norm_name == 'batch':
            norm_name = 'tbatch'
    if model in ['cnn', 'resnet']:
        if norm_name in ['layer', 'batch']:
            norm_name = f'{norm_name}2d'
            
        
    normalizations = {
        'batch': nn.BatchNorm1d(num_features, affine=affine),
        'tbatch': TransformerBatchNorm(num_features, affine=affine),
        'batch2d': nn.BatchNorm2d(num_features, affine=affine),
        'layer': nn.LayerNorm(num_features, elementwise_affine=affine),
        'layer2d': nn.GroupNorm(1, num_features, affine=affine),
        # 'layer2d': nn.LayerNorm([num_features, spatial_size, spatial_size], elementwise_affine=affine),  # Custom 2D layer norm for CNN/ResNet
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
