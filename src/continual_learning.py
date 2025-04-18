"""
Combined file with neural network models, NetworkMonitor for tracking activations and gradients,
and utilities for metrics computation and continual learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
import wandb  # For logging experiments

###########################################
# Utility Functions and Custom Layers
###########################################

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

###########################################
# Model Definitions
###########################################

class MLP(nn.Module):
    def __init__(self, 
                 input_size=784, 
                 hidden_sizes=[512, 256, 128], 
                 output_size=10, 
                 activation='relu',
                 dropout_p=0.0,
                 normalization=None,
                 norm_after_activation=False,
                 bias=True,
                 normalization_affine=True):
        """Fully connected MLP with customizable architecture."""
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.norm_after_activation = norm_after_activation
        
        self.layers = nn.ModuleDict()
        in_features = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            self.layers[f'linear_{i}'] = nn.Linear(in_features, hidden_size, bias=bias)
            
            if norm_after_activation:
                self.layers[f'act_{i}'] = get_activation(activation)
                if normalization:
                    self.layers[f'norm_{i}'] = get_normalization(normalization, hidden_size, affine=normalization_affine)
            else:
                if normalization:
                    self.layers[f'norm_{i}'] = get_normalization(normalization, hidden_size, affine=normalization_affine)
                self.layers[f'act_{i}'] = get_activation(activation)
            
            if dropout_p > 0:
                self.layers[f'drop_{i}'] = nn.Dropout(dropout_p)
            
            in_features = hidden_size
        
        self.layers['out'] = nn.Linear(in_features, output_size, bias=bias)
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        for k, l in self.layers.items():
            x = l(x)
        
        return x

class CNN(nn.Module):
    def __init__(self, 
                 in_channels=3,
                 conv_channels=[64, 128, 256], 
                 kernel_sizes=[3, 3, 3],
                 strides=[1, 1, 1],
                 paddings=[1, 1, 1],
                 fc_hidden_units=[512],
                 num_classes=10, 
                 input_size=32,
                 activation='relu',
                 dropout_p=0.0,
                 pool_type='max',
                 pool_size=2,
                 use_batchnorm=True,
                 norm_after_activation=False,
                 normalization_affine=True):
        """CNN with configurable layers, activations, and normalizations."""
        super(CNN, self).__init__()
        
        assert len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "Convolutional parameters must have the same length"
        
        self.norm_after_activation = norm_after_activation
        
        self.layers = nn.ModuleDict()
        
        channels = in_channels
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(conv_channels, kernel_sizes, strides, paddings)):
            self.layers[f'conv_{i}'] = nn.Conv2d(channels, out_channels, kernel_size, stride, padding)
            
            if use_batchnorm:
                self.layers[f'norm_{i}'] = get_normalization('batch2d', out_channels, affine=normalization_affine)
            
            self.layers[f'act_{i}'] = get_activation(activation)
            
            if pool_type == 'max':
                self.layers[f'pool_{i}'] = nn.MaxPool2d(pool_size, pool_size)
            elif pool_type == 'avg':
                self.layers[f'pool_{i}'] = nn.AvgPool2d(pool_size, pool_size)
            
            channels = out_channels
        
        num_pools = len(conv_channels) if pool_type in ['max', 'avg'] else 0
        final_size = input_size // (pool_size ** num_pools)
        self.flattened_size = conv_channels[-1] * final_size * final_size
        
        self.layers['flatten'] = nn.Flatten()
        
        # Build fully connected layers
        fc_input_size = self.flattened_size
        for i, hidden_units in enumerate(fc_hidden_units):
            self.layers[f'fc_{i}'] = nn.Linear(fc_input_size, hidden_units)
            self.layers[f'fc_act_{i}'] = get_activation(activation)
            
            if dropout_p > 0:
                self.layers[f'fc_drop_{i}'] = nn.Dropout(dropout_p)
            
            fc_input_size = hidden_units
        
        self.layers['fc_out'] = nn.Linear(fc_input_size, num_classes)
    
    def forward(self, x):
        for k, l in self.layers.items():
            x = l(x)
        return x


class BasicBlock(nn.Module):
    """Basic ResNet block with activation and normalization."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu', 
                 use_batchnorm=True, norm_after_activation=False, downsample=None,
                 normalization_affine=True):
        super(BasicBlock, self).__init__()
        
        self.norm_after_activation = norm_after_activation
        self.layers = nn.ModuleDict()
        
        self.layers['conv1'] = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                                        padding=1, bias=not use_batchnorm)
        
        if use_batchnorm:
            self.layers['bn1'] = get_normalization('batch2d', planes, affine=normalization_affine)
        
        self.layers['activation'] = get_activation(activation)
        
        self.layers['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                                        padding=1, bias=not use_batchnorm)
        
        if use_batchnorm:
            self.layers['bn2'] = get_normalization('batch2d', planes, affine=normalization_affine)
        
        if downsample is not None:
            self.layers['downsample'] = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.layers['conv1'](x)
        
        if 'bn1' in self.layers and not self.norm_after_activation:
            out = self.layers['bn1'](out)
        
        out = self.layers['activation'](out)
        
        if 'bn1' in self.layers and self.norm_after_activation:
            out = self.layers['bn1'](out)
            
        out = self.layers['conv2'](out)
        
        if 'bn2' in self.layers and not self.norm_after_activation:
            out = self.layers['bn2'](out)
            
        if 'downsample' in self.layers:
            identity = self.layers['downsample'](x)
            
        out = out + identity
        out = self.layers['activation'](out)
        
        if 'bn2' in self.layers and self.norm_after_activation:
            out = self.layers['bn2'](out)
            
        return out


class ResNet(nn.Module):
    """ResNet architecture for continual learning experiments."""
    def __init__(self, 
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],
                 num_classes=10,
                 in_channels=3,
                 base_channels=64,
                 activation='relu',
                 dropout_p=0.0,
                 use_batchnorm=True,
                 norm_after_activation=False,
                 normalization_affine=True):
        super(ResNet, self).__init__()
        
        self.use_batchnorm = use_batchnorm
        self.norm_after_activation = norm_after_activation
        self.in_planes = base_channels
        
        self.layers = nn.ModuleDict()
        
        self.layers['conv1'] = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                                        stride=1, padding=1, bias=not use_batchnorm)
        
        if use_batchnorm:
            self.layers['bn1'] = get_normalization('batch2d', base_channels, affine=normalization_affine)
        
        self.layers['activation'] = get_activation(activation)
        
        # Create ResNet blocks
        self._make_layer(block, base_channels, layers[0], stride=1, 
                        activation=activation, use_batchnorm=use_batchnorm, 
                        norm_after_activation=norm_after_activation, 
                        layer_name='layer1',
                        normalization_affine=normalization_affine)
        self._make_layer(block, base_channels*2, layers[1], stride=2, 
                        activation=activation, use_batchnorm=use_batchnorm, 
                        norm_after_activation=norm_after_activation, 
                        layer_name='layer2',
                        normalization_affine=normalization_affine)
        self._make_layer(block, base_channels*4, layers[2], stride=2, 
                        activation=activation, use_batchnorm=use_batchnorm,
                        norm_after_activation=norm_after_activation, 
                        layer_name='layer3',
                        normalization_affine=normalization_affine)
        self._make_layer(block, base_channels*8, layers[3], stride=2, 
                        activation=activation, use_batchnorm=use_batchnorm,
                        norm_after_activation=norm_after_activation, 
                        layer_name='layer4',
                        normalization_affine=normalization_affine)
        
        self.layers['avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
        self.layers['flatten'] = nn.Flatten()
        
        if dropout_p > 0:
            self.layers['dropout'] = nn.Dropout(dropout_p)
        
        self.layers['fc'] = nn.Linear(base_channels*8*block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.num_layers = len(layers)
        self.blocks_per_layer = layers
                
    def _make_layer(self, block, planes, num_blocks, stride=1, activation='relu', 
                    use_batchnorm=True, norm_after_activation=False, layer_name='layer',
                    normalization_affine=True):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample_layers = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, 
                         kernel_size=1, stride=stride, bias=not use_batchnorm)
            )
            
            if use_batchnorm:
                downsample_layers.add_module('1', get_normalization('batch2d', planes * block.expansion, affine=normalization_affine))
                
            downsample = downsample_layers
        
        self.layers[f'{layer_name}_block0'] = block(
            self.in_planes, planes, stride, activation, 
            use_batchnorm, norm_after_activation, downsample,
            normalization_affine=normalization_affine
        )
        
        self.in_planes = planes * block.expansion
        
        for i in range(1, num_blocks):
            self.layers[f'{layer_name}_block{i}'] = block(
                self.in_planes, planes, 1, activation, 
                use_batchnorm, norm_after_activation,
                normalization_affine=normalization_affine
            )
        
    def forward(self, x):
        x = self.layers['conv1'](x)
        
        if self.use_batchnorm and not self.norm_after_activation:
            if 'bn1' in self.layers:
                x = self.layers['bn1'](x)
                
        x = self.layers['activation'](x)
        
        if self.use_batchnorm and self.norm_after_activation:
            if 'bn1' in self.layers:
                x = self.layers['bn1'](x)
        
        # Forward through ResNet blocks
        for layer_idx in range(1, self.num_layers + 1):
            for block_idx in range(self.blocks_per_layer[layer_idx - 1]):
                block_name = f'layer{layer_idx}_block{block_idx}'
                x = self.layers[block_name](x)
        
        x = self.layers['avgpool'](x)
        x = self.layers['flatten'](x)
        
        if 'dropout' in self.layers:
            x = self.layers['dropout'](x)
            
        x = self.layers['fc'](x)
            
        return x


###########################################
# ViT Components
###########################################

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5

        self.layers = nn.ModuleDict({
            'qkv': nn.Linear(dim, dim * 3, bias=qkv_bias),
            'attn_drop': nn.Dropout(attn_drop),
            'proj': nn.Linear(dim, dim),
            'proj_drop': nn.Dropout(proj_drop)
        })

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.layers['qkv'](x).reshape(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.layers['attn_drop'](attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.layers['proj'](x)
        x = self.layers['proj_drop'](x)
        return x


class TransformerMLP(nn.Module):
    """MLP module with activation."""
    def __init__(self, in_features, hidden_features, out_features, 
                 activation='gelu', drop=0.):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'fc1': nn.Linear(in_features, hidden_features),
            'act': get_activation(activation),
            'drop1': nn.Dropout(drop) if drop > 0 else nn.Identity(),
            'fc2': nn.Linear(hidden_features, out_features),
            'drop2': nn.Dropout(drop) if drop > 0 else nn.Identity()
        })

    def forward(self, x):
        x = self.layers['fc1'](x)
        x = self.layers['act'](x)
        x = self.layers['drop1'](x)
        x = self.layers['fc2'](x)
        x = self.layers['drop2'](x)
        return x


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding for Vision Transformer."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.layers = nn.ModuleDict({
            'proj': nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
        })

    def forward(self, x):
        x = self.layers['proj'](x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with components."""
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, drop=0., 
                 attn_drop=0., activation='gelu', normalization='layer',
                 normalization_affine=True):
        super().__init__()
        
        self.layers = nn.ModuleDict({
            'norm1': get_normalization(normalization, dim, affine=normalization_affine),
            'attn': Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, 
                             attn_drop=attn_drop, proj_drop=drop),
            'norm2': get_normalization(normalization, dim, affine=normalization_affine),
            'mlp': TransformerMLP(dim, int(dim * mlp_ratio), dim, 
                       activation=activation, drop=drop)
        })

    def forward(self, x):
        norm_x = self.layers['norm1'](x)
        attn_out = self.layers['attn'](norm_x)
        x = x + attn_out
        
        norm_x = self.layers['norm2'](x)
        mlp_out = self.layers['mlp'](norm_x)
        x = x + mlp_out
            
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model."""
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 in_channels=3, 
                 num_classes=10, 
                 embed_dim=192,
                 depth=12, 
                 n_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop_rate=0.1,
                 attn_drop_rate=0.0,
                 activation='gelu',
                 normalization='layer',
                 normalization_affine=True):
        super().__init__()
        
        self.layers = nn.ModuleDict()
        
        self.layers['patch_embed'] = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.layers['patch_embed'].n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.layers['pos_drop'] = nn.Dropout(drop_rate)

        for i in range(depth):
            self.layers[f'block_{i}'] = TransformerBlock(
                embed_dim, n_heads, mlp_ratio, qkv_bias, 
                drop_rate, attn_drop_rate, activation, normalization,
                normalization_affine=normalization_affine
            )

        self.layers['norm'] = get_normalization(normalization, embed_dim, affine=normalization_affine)
        self.layers['head'] = nn.Linear(embed_dim, num_classes)

        self._init_weights()
        self.depth = depth

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.layers['patch_embed'](x)
        
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        x = x + self.pos_embed
        x = self.layers['pos_drop'](x)

        for i in range(self.depth):
            x = self.layers[f'block_{i}'](x)
        
        x = self.layers['norm'](x)
        x = x[:, 0]  # Use CLS token for classification
        x = self.layers['head'](x)
            
        return x


###########################################
# NetworkMonitor Class
###########################################

class NetworkMonitor:
    def __init__(self, model, filter_func=None):
        """
        Initialize the network monitor.
        
        Args:
            model: The neural network model to monitor
            filter_func: Function that takes a layer name and returns 
                         True if the layer should be monitored
        """
        self.model = model
        self.filter_func = filter_func if filter_func is not None else lambda name: True
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        self.fwd_hooks = []
        self.bwd_hooks = []
        self.hooks_active = False
        
    def set_filter(self, filter_func):
        """Update the filter function for selecting layers to monitor."""
        was_active = self.hooks_active
        if was_active:
            self.remove_hooks()
        self.filter_func = filter_func if filter_func is not None else lambda name: True
        if was_active:
            self.register_hooks()
        
    def register_hooks(self):
        """Register forward and backward hooks on the model."""
        if not self.hooks_active:
            for name, module in self.model.named_modules():
                if name != '' and self.filter_func(name):
                    def make_fwd_hook(name=name):
                        def hook(module, input, output):
                            self.activations[f"{name}"].append(output.clone().detach().cpu())
                        return hook
                    
                    def make_bwd_hook(name=name):
                        def hook(module, grad_input, grad_output):
                            if len(grad_output) > 0 and grad_output[0] is not None:
                                self.gradients[f"{name}"].append(grad_output[0].clone().detach().cpu())
                            return grad_input
                        return hook
                    
                    h1 = module.register_forward_hook(make_fwd_hook())
                    h2 = module.register_full_backward_hook(make_bwd_hook())
                    self.fwd_hooks.append(h1)
                    self.bwd_hooks.append(h2)
            
            self.hooks_active = True
    
    def remove_hooks(self):
        """Remove all hooks from the model."""
        if self.hooks_active:
            for h in self.fwd_hooks + self.bwd_hooks:
                h.remove()
            self.fwd_hooks = []
            self.bwd_hooks = []
            self.hooks_active = False
        
    def clear_data(self):
        """Clear stored activations and gradients."""
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        
    def get_latest_activations(self):
        """Get the latest activations for all monitored layers."""
        latest_acts = {}
        for name, acts_list in self.activations.items():
            if acts_list:
                latest_acts[name] = acts_list[-1]
        return latest_acts
    
    def get_latest_gradients(self):
        """Get the latest gradients for all monitored layers."""
        latest_grads = {}
        for name, grads_list in self.gradients.items():
            if grads_list:
                latest_grads[name] = grads_list[-1]
        return latest_grads


###########################################
# Metric Functions 
###########################################

def flatten_activations(layer_act):
    """Reshape layer activations to 2D matrix (samples × features)."""
    shape = layer_act.shape
    if len(shape) == 4:  # Convolutional layer
        return layer_act.permute(0, 2, 3, 1).contiguous().view(-1, shape[1])
    elif len(shape) == 3:  # Transformer layer
        return layer_act.contiguous().view(-1, shape[2])
    else:  # Linear layer
        return layer_act.view(-1, shape[1])

def measure_dead_neurons(layer_act, dead_threshold=0.95):
    """Measure fraction of neurons that are inactive (dead)."""
    flattened_act = flatten_activations(layer_act)
    is_zero = (flattened_act.abs() < 1e-7)
    frac_zero_per_neuron = is_zero.float().mean(dim=0)
    dead_mask = (frac_zero_per_neuron > dead_threshold)
    dead_fraction = dead_mask.float().mean().item()
    return dead_fraction

def measure_duplicate_neurons(layer_act, corr_threshold):
    """Measure fraction of neurons that are duplicates of others."""
    flattened_act = flatten_activations(layer_act)
    flattened_act = flattened_act.t()  
    flattened_act = torch.nn.functional.normalize(flattened_act, p=2, dim=1)
    similarity_matrix = torch.matmul(flattened_act, flattened_act.t())
    upper_tri_mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    dup_pairs = (similarity_matrix > corr_threshold) & upper_tri_mask
    neuron_is_dup = dup_pairs.any(dim=1)
    fraction_dup = neuron_is_dup.float().mean().item()
    return fraction_dup

def measure_effective_rank(layer_act, svd_sample_size=1024):
    """Compute effective rank (entropy of normalized singular values)."""
    flattened_act = flatten_activations(layer_act)
    N = flattened_act.shape[0]
    if N > svd_sample_size:
        idx = torch.randperm(N)[:svd_sample_size]
        flattened_act = flattened_act[idx]
    U, S, Vt = torch.linalg.svd(flattened_act, full_matrices=False)
    S_sum = S.sum()
    if S_sum < 1e-9:
        return 0.0
    p = S / S_sum
    p_log_p = p * torch.log(p + 1e-12)
    eff_rank = torch.exp(-p_log_p.sum()).item()
    return eff_rank

def measure_stable_rank(layer_act, sample_size=1024, use_gram=True):
    """Compute stable rank (squared Frobenius norm / spectral norm squared)."""
    flattened_act = flatten_activations(layer_act)
    N, D = flattened_act.shape
    if N > sample_size:
        idx = torch.randperm(N)[:sample_size]
        flattened_act = flattened_act[idx]
        N = sample_size
    flattened_act = flattened_act - flattened_act.mean(dim=0, keepdim=True)
    if use_gram or D < N:
        frob_norm_sq = torch.sum(flattened_act**2).item()
        gram = torch.matmul(flattened_act.t(), flattened_act)
        trace_gram_squared = torch.sum(gram**2).item()
        if trace_gram_squared < 1e-9:
            return 0.0
        stable_rank = (frob_norm_sq**2) / trace_gram_squared
    else:
        cov = torch.matmul(flattened_act, flattened_act.t())
        trace_cov = torch.trace(cov).item()
        trace_cov_squared = torch.sum(cov**2).item()
        if trace_cov_squared < 1e-9:
            return 0.0
        stable_rank = (trace_cov**2) / trace_cov_squared
    return stable_rank

def measure_saturated_neurons(layer_act, layer_grad, saturation_threshold=1e-4, saturation_percentage=0.99):
    """
    Measures the fraction of saturated neurons in a layer.
    
    Saturated neurons are identified as those where the ratio of gradient magnitude
    to mean activation magnitude is very small, indicating the neuron is in a flat
    region of the loss landscape.
    """
    flattened_act = flatten_activations(layer_act)
    flattened_grad = flatten_activations(layer_grad)
    
    # Calculate the mean activation magnitude for each neuron
    mean_act_magnitude = flattened_act.abs().mean(dim=0, keepdim=True)
    
    # Avoid division by zero
    mean_act_magnitude = torch.clamp(mean_act_magnitude, min=1e-12)
    
    # Calculate the ratio of gradient magnitude to mean activation magnitude
    saturation_ratio = flattened_grad.abs() / mean_act_magnitude
    
    # Mark neurons as saturated if the ratio is below the threshold
    is_saturated = (saturation_ratio < saturation_threshold).float()
    
    # Calculate fraction of samples where each neuron appears saturated
    saturation_per_neuron = is_saturated.mean(dim=0)
    
    # Consider a neuron truly saturated if it's saturated in most samples
    saturated_mask = (saturation_per_neuron > saturation_percentage)
    
    # Calculate the overall fraction of saturated neurons
    saturated_fraction = saturated_mask.float().mean().item()
    
    return saturated_fraction


###########################################
# Analysis with Monitor
###########################################

def analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, 
                        dead_threshold, 
                        corr_threshold, 
                        saturation_threshold, 
                        saturation_percentage,
                        device='cpu'):
    """
    Analyze model behavior on a fixed batch to compute metrics.
    
    Args:
        model: Neural network model
        monitor: NetworkMonitor instance
        fixed_batch: Input data batch
        fixed_targets: Target labels
        criterion: Loss function
        dead_threshold: Threshold for dead neuron detection
        corr_threshold: Threshold for duplicate neuron detection
        saturation_threshold: Threshold for saturated neuron detection
        saturation_percentage: Percentage of samples required for a neuron to be considered saturated
        device: Device to run computations on
        
    Returns:
        Dictionary of metrics for each layer
    """
    if fixed_batch.device != device:
        fixed_batch = fixed_batch.to(device)
        fixed_targets = fixed_targets.to(device)
    
    hooks_were_active = monitor.hooks_active
    monitor.register_hooks()
    
    with torch.set_grad_enabled(criterion is not None):
        outputs = model(fixed_batch)
        loss = criterion(outputs, fixed_targets)
        loss.backward()
    
    metrics = {}
    latest_acts = monitor.get_latest_activations()
    latest_grads = monitor.get_latest_gradients()

    for layer_name, act in latest_acts.items():
        # Skip layers without gradients when computing metrics
        if layer_name not in latest_grads:
            continue
            
        grad = latest_grads[layer_name]
        
        # Compute all metrics for this layer
        metrics[layer_name] = {
            'dead_fraction': measure_dead_neurons(act, dead_threshold),
            'dup_fraction': measure_duplicate_neurons(act, corr_threshold),
            'eff_rank': measure_effective_rank(act),
            'stable_rank': measure_stable_rank(act),
            'saturated_frac': measure_saturated_neurons(act, grad, saturation_threshold, saturation_percentage),
        }
    
    if not hooks_were_active:
        monitor.remove_hooks()
    
    return metrics


###########################################
# Continual Learning Dataset Functions
###########################################

class SubsetDataset(Dataset):
    """Dataset wrapper for class subset selection"""
    def __init__(self, dataset, class_indices):
        self.dataset = dataset
        self.class_indices = class_indices
        self.indices = self._get_indices()
        
    def _get_indices(self):
        indices = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if label in self.class_indices:
                indices.append(i)
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return image, label


def prepare_continual_learning_data(dataset, class_sequence, batch_size=128, val_split=0.2):
    """
    Prepare dataloaders for continual learning on a sequence of class subsets.
    
    Args:
        dataset: The full dataset (e.g., CIFAR10)
        class_sequence: List of lists, where each inner list contains class indices for a task
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
    
    Returns:
        Dictionary mapping task_id -> (train_loader, val_loader, fixed_train_loader, fixed_val_loader)
    """
    dataloaders = {}
    all_seen_classes = set()
    
    for task_id, classes in enumerate(class_sequence):
        current_classes = set(classes)
        
        # Create current task dataset
        current_dataset = SubsetDataset(dataset, classes)
        
        # Split into training and validation
        dataset_size = len(current_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_subset = Subset(current_dataset, train_indices)
        val_subset = Subset(current_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size, shuffle=False)
        
        # For previous tasks (old classes)
        old_loaders = {}
        if task_id > 0:
            old_classes = all_seen_classes - current_classes
            if old_classes:
                old_dataset = SubsetDataset(dataset, list(old_classes))
                old_size = len(old_dataset)
                old_indices = list(range(old_size))
                random.shuffle(old_indices)
                
                old_train_size = int((1 - val_split) * old_size)
                old_train_indices = old_indices[:old_train_size]
                old_val_indices = old_indices[old_train_size:]
                
                old_train_subset = Subset(old_dataset, old_train_indices)
                old_val_subset = Subset(old_dataset, old_val_indices)
                
                old_train_loader = DataLoader(old_train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
                old_val_loader = DataLoader(old_val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
                
                # Fixed old batches for metrics
                fixed_old_train = Subset(old_train_subset, range(min(500, len(old_train_subset))))
                fixed_old_val = Subset(old_val_subset, range(min(500, len(old_val_subset))))
                
                fixed_old_train_loader = DataLoader(fixed_old_train, batch_size=batch_size, shuffle=False)
                fixed_old_val_loader = DataLoader(fixed_old_val, batch_size=batch_size, shuffle=False)
                
                old_loaders = {
                    'train': old_train_loader,
                    'val': old_val_loader,
                    'fixed_train': fixed_old_train_loader,
                    'fixed_val': fixed_old_val_loader
                }
        
        # Store the dataloaders for this task
        dataloaders[task_id] = {
            'current': {
                'train': train_loader,
                'val': val_loader,
                'fixed_train': fixed_train_loader,
                'fixed_val': fixed_val_loader,
                'classes': classes
            },
            'old': old_loaders
        }
        
        # Update the set of all seen classes
        all_seen_classes.update(current_classes)
    
    return dataloaders


###########################################
# Training Functions
###########################################


def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    Evaluate model on a dataset.
    
    Returns:
        loss, accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total


###########################################
# Continual Learning Training Functions
###########################################

def train_continual_learning(model, 
                             task_dataloaders, 
                             config, 
                             device='cpu'):
    """
    Train a model using continual learning on a sequence of tasks.
    
    Args:
        model: The neural network model
        task_dataloaders: Dictionary mapping task_id -> task data loaders
        config: Configuration dictionary
        device: Device to train on
    
    Returns:
        Dictionary with training history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    dead_threshold=config['dead_threshold'] 
    corr_threshold=config['corr_threshold'] 
    saturation_threshold=config['saturation_threshold'] 
    saturation_percentage=config['saturation_percentage']

    # Create module filter function
    def module_filter(name):
        return 'linear' in name or '.mlp' in name or 'fc' in name or name.endswith('.proj')
    
    # For monitoring metrics
    train_monitor = NetworkMonitor(model, module_filter)
    val_monitor = NetworkMonitor(model, module_filter)
    
    # History tracking
    history = {
        'tasks': {}
    }
    
    print(f"Starting continual learning with {len(task_dataloaders)} tasks...")


    
    def analyze_callback(monitor, fixed_batch, fixed_targets,):
        return analyze_fixed_batch(model, monitor, fixed_batch, fixed_targets, criterion, device=device, 
                                   dead_threshold=dead_threshold, corr_threshold=corr_threshold, 
                                   saturation_threshold=saturation_threshold, saturation_percentage=saturation_percentage,)
    def analyze_train_callback():
        return analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)

    def analyze_val_callback():
        return analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
    
    for task_id, task_data in task_dataloaders.items():
        print(f"\n{'='*50}")
        print(f"Starting Task {task_id}: Classes {task_data['current']['classes']}")
        print(f"{'='*50}")
        
        current_train_loader = task_data['current']['train']
        current_val_loader = task_data['current']['val']
        current_fixed_train = task_data['current']['fixed_train']
        current_fixed_val = task_data['current']['fixed_val']
        
        task_history = {
            'epochs': [],
            'current': {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            },
            'old': {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            },
            'training_metrics_history': defaultdict(lambda: defaultdict(list)),
            'validation_metrics_history': defaultdict(lambda: defaultdict(list))
        }
        
        # Get a fixed batch for metrics
        try:
            fixed_train_batch, fixed_train_targets = next(iter(current_fixed_train))
            fixed_val_batch, fixed_val_targets = next(iter(current_fixed_val))
            
            fixed_train_batch = fixed_train_batch.to(device)
            fixed_train_targets = fixed_train_targets.to(device)
            fixed_val_batch = fixed_val_batch.to(device)
            fixed_val_targets = fixed_val_targets.to(device)
            
            # Initial metrics
            print("Measuring initial metrics...")
            
            train_metrics = analyze_train_callback()
            val_metrics = analyze_val_callback()
            
            for layer_name, metrics in train_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['training_metrics_history'][layer_name][metric_name].append(value)
            
            for layer_name, metrics in val_metrics.items():
                for metric_name, value in metrics.items():
                    task_history['validation_metrics_history'][layer_name][metric_name].append(value)
        except StopIteration:
            print("Warning: Not enough samples for fixed batch metrics")
        
        # If there are old classes (previous tasks)
        has_old_data = 'old' in task_data and task_data['old']
        if has_old_data:
            old_train_loader = task_data['old']['train']
            old_val_loader = task_data['old']['val']
            
            # Evaluate on old data before training on new data
            old_train_loss, old_train_acc = evaluate_model(model, old_train_loader, criterion, device)
            old_val_loss, old_val_acc = evaluate_model(model, old_val_loader, criterion, device)
            
            print(f"Initial performance on OLD classes:")
            print(f"  Train Loss: {old_train_loss:.4f}, Train Acc: {old_train_acc:.2f}%")
            print(f"  Val Loss: {old_val_loss:.4f}, Val Acc: {old_val_acc:.2f}%")
        
        # Evaluate on current task before training
        current_train_loss, current_train_acc = evaluate_model(model, current_train_loader, criterion, device)
        current_val_loss, current_val_acc = evaluate_model(model, current_val_loader, criterion, device)
        
        print(f"Initial performance on CURRENT classes:")
        print(f"  Train Loss: {current_train_loss:.4f}, Train Acc: {current_train_acc:.2f}%")
        print(f"  Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.2f}%")
        
        # Record initial metrics
        task_history['epochs'].append(0)
        task_history['current']['train_loss'].append(current_train_loss)
        task_history['current']['train_acc'].append(current_train_acc)
        task_history['current']['val_loss'].append(current_val_loss)
        task_history['current']['val_acc'].append(current_val_acc)
        
        if has_old_data:
            task_history['old']['train_loss'].append(old_train_loss)
            task_history['old']['train_acc'].append(old_train_acc)
            task_history['old']['val_loss'].append(old_val_loss)
            task_history['old']['val_acc'].append(old_val_acc)
        
        # Training loop for this task
        start_time = time.time()
        for epoch in range(1, config["epochs_per_task"] + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in current_train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            epoch_train_loss = running_loss / len(current_train_loader)
            epoch_train_acc = 100. * correct / total
            
            # Evaluate on current task
            current_val_loss, current_val_acc = evaluate_model(model, current_val_loader, criterion, device)
            
            # Record current task metrics
            task_history['epochs'].append(epoch)
            task_history['current']['train_loss'].append(epoch_train_loss)
            task_history['current']['train_acc'].append(epoch_train_acc)
            task_history['current']['val_loss'].append(current_val_loss)
            task_history['current']['val_acc'].append(current_val_acc)
            
            # If there are old classes, evaluate on them too
            if has_old_data:
                old_train_loss, old_train_acc = evaluate_model(model, old_train_loader, criterion, device)
                old_val_loss, old_val_acc = evaluate_model(model, old_val_loader, criterion, device)
                
                task_history['old']['train_loss'].append(old_train_loss)
                task_history['old']['train_acc'].append(old_train_acc)
                task_history['old']['val_loss'].append(old_val_loss)
                task_history['old']['val_acc'].append(old_val_acc)
            
            # Periodically collect network metrics
            if epoch % config["metrics_frequency"] == 0 or epoch == config["epochs_per_task"]:
                try:
                    train_monitor.clear_data()
                    val_monitor.clear_data()
                    
                    train_metrics = analyze_callback(train_monitor, fixed_train_batch, fixed_train_targets)
                    val_metrics = analyze_callback(val_monitor, fixed_val_batch, fixed_val_targets)
                    
                    # Log current fixed batch metrics to wandb
                    fixed_metrics_log = {"task": task_id, "epoch": epoch}
                    for layer_name, metrics in train_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"current_fixed_train/{layer_name}/{metric_name}"] = value
                    for layer_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            fixed_metrics_log[f"current_fixed_val/{layer_name}/{metric_name}"] = value
                    
                    # Log old fixed batch metrics if available
                    if has_old_data and 'fixed_train' in task_data['old'] and 'fixed_val' in task_data['old']:
                        try:
                            old_fixed_train = task_data['old']['fixed_train']
                            old_fixed_val = task_data['old']['fixed_val']
                            
                            old_fixed_train_batch, old_fixed_train_targets = next(iter(old_fixed_train))
                            old_fixed_val_batch, old_fixed_val_targets = next(iter(old_fixed_val))
                            
                            old_fixed_train_batch = old_fixed_train_batch.to(device)
                            old_fixed_train_targets = old_fixed_train_targets.to(device)
                            old_fixed_val_batch = old_fixed_val_batch.to(device)
                            old_fixed_val_targets = old_fixed_val_targets.to(device)
                            
                            # Clear monitors
                            train_monitor.clear_data()
                            val_monitor.clear_data()
                            
                            old_train_metrics = analyze_callback(train_monitor, old_fixed_train_batch, old_fixed_train_targets)
                            old_val_metrics = analyze_callback(val_monitor, old_fixed_val_batch, old_fixed_val_targets)
                            
                            # Log old fixed batch metrics to wandb
                            for layer_name, metrics in old_train_metrics.items():
                                for metric_name, value in metrics.items():
                                    fixed_metrics_log[f"old_fixed_train/{layer_name}/{metric_name}"] = value
                            for layer_name, metrics in old_val_metrics.items():
                                for metric_name, value in metrics.items():
                                    fixed_metrics_log[f"old_fixed_val/{layer_name}/{metric_name}"] = value
                        except Exception as e:
                            print(f"Error collecting old fixed batch metrics: {e}")
                    
                    # Log all metrics to wandb
                    wandb.log(fixed_metrics_log)
                    
                    # Store metrics in history for later analysis
                    for layer_name, metrics in train_metrics.items():
                        for metric_name, value in metrics.items():
                            task_history['training_metrics_history'][layer_name][metric_name].append(value)
                    
                    for layer_name, metrics in val_metrics.items():
                        for metric_name, value in metrics.items():
                            task_history['validation_metrics_history'][layer_name][metric_name].append(value)
                except Exception as e:
                    print(f"Error collecting metrics: {e}")
            
            # Log to wandb
            log_data = {
                "task": task_id,
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": current_val_loss,
                "val_acc": current_val_acc
            }
            
            if has_old_data:
                log_data.update({
                    "old_train_loss": old_train_loss,
                    "old_train_acc": old_train_acc,
                    "old_val_loss": old_val_loss,
                    "old_val_acc": old_val_acc
                })
            
            wandb.log(log_data)
            
            # Print progress
            elapsed = time.time() - start_time
            print(f'Task {task_id}, Epoch {epoch}/{config["epochs_per_task"]}:')
            print(f'  CURRENT: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
                 f'Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.2f}%')
            
            if has_old_data:
                print(f'  OLD: Train Loss: {old_train_loss:.4f}, Train Acc: {old_train_acc:.2f}%, '
                     f'Val Loss: {old_val_loss:.4f}, Val Acc: {old_val_acc:.2f}%')
            
            print(f'  Time: {elapsed:.2f}s')
        
        # Store task history
        history['tasks'][task_id] = {
            'classes': task_data['current']['classes'],
            'history': task_history
        }
        
    
    return history

