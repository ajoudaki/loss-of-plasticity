import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

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
