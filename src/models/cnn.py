import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

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