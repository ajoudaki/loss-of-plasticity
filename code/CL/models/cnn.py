"""
Customizable CNN model for continual learning experiments.
"""

import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

class ConfigurableCNN(nn.Module):
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
                 record_activations=False):
        """
        Customizable CNN with configurable layers, activations, and normalizations.
        
        Parameters:
            in_channels (int): Number of input channels (3 for RGB images)
            conv_channels (list): List of convolutional layer output channels
            kernel_sizes (list): List of kernel sizes for each conv layer
            strides (list): List of stride values for each conv layer
            paddings (list): List of padding values for each conv layer
            fc_hidden_units (list): List of hidden units for fully connected layers
            num_classes (int): Number of output classes
            input_size (int): Height/width of the input images (assumed square)
            activation (str): Activation function to use ('relu', 'tanh', etc.)
            dropout_p (float): Dropout probability (0 to disable)
            pool_type (str): Type of pooling ('max', 'avg', or None)
            pool_size (int): Size of the pooling window
            use_batchnorm (bool): Whether to use batch normalization
            norm_after_activation (bool): Apply normalization after activation
            record_activations (bool): Whether to store activations for analysis
        """
        super(ConfigurableCNN, self).__init__()
        
        self.record_activations = record_activations
        
        # Check if input lists are of the same length
        assert len(conv_channels) == len(kernel_sizes) == len(strides) == len(paddings), \
            "Convolutional parameters (channels, kernels, strides, paddings) must have the same length"
        
        # Conv layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        
        channels = in_channels
        
        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(conv_channels, kernel_sizes, strides, paddings)):
            # Conv layer
            conv = nn.Conv2d(channels, out_channels, kernel_size, stride, padding)
            self.conv_layers.append(conv)
            
            # Normalization
            if use_batchnorm:
                norm = nn.BatchNorm2d(out_channels)
                self.norm_layers.append(norm)
            else:
                self.norm_layers.append(None)
            
            # Activation
            act = get_activation(activation)
            self.activation_layers.append(act)
            
            channels = out_channels
        
        # Pooling
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(pool_size, pool_size)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(pool_size, pool_size)
        else:
            self.pool = nn.Identity()
        
        # Calculate the size after all pooling operations
        num_pools = len(conv_channels) if pool_type in ['max', 'avg'] else 0
        final_size = input_size // (pool_size ** num_pools)
        self.flattened_size = conv_channels[-1] * final_size * final_size
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_activations = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        fc_input_size = self.flattened_size
        for hidden_units in fc_hidden_units:
            self.fc_layers.append(nn.Linear(fc_input_size, hidden_units))
            self.fc_activations.append(get_activation(activation))
            self.fc_dropouts.append(nn.Dropout(dropout_p) if dropout_p > 0 else None)
            fc_input_size = hidden_units
        
        # Output layer
        self.fc_output = nn.Linear(fc_input_size, num_classes)
        
        # Activation storage
        self.stored_activations = {}
    
    def forward(self, x, store_activations=False):
        """
        Forward pass with optional activation storage.
        
        Parameters:
            x (torch.Tensor): Input data [batch_size, in_channels, height, width]
            store_activations (bool): Whether to store activations from this pass
        
        Returns:
            torch.Tensor: Output logits
            dict (optional): Hidden activations if record_activations=True
        """
        # Should we store activations for this pass?
        should_store = store_activations or self.record_activations
        activations = {} if should_store else None
        
        if should_store:
            activations['input'] = x
        
        # Conv layers
        for i, (conv, norm, act) in enumerate(zip(self.conv_layers, self.norm_layers, self.activation_layers)):
            x = conv(x)
            if should_store:
                activations[f'conv_{i}'] = x.detach().clone()
            
            if norm and not self.norm_after_activation:
                x = norm(x)
                
            x = act(x)
            if should_store:
                activations[f'conv_act_{i}'] = x.detach().clone()
                
            if norm and self.norm_after_activation:
                x = norm(x)
                
            x = self.pool(x)
            if should_store:
                activations[f'pool_{i}'] = x.detach().clone()
        
        # Flatten
        x = x.view(x.size(0), -1)
        if should_store:
            activations['flatten'] = x.detach().clone()
        
        # FC layers
        for i, (fc, act, dropout) in enumerate(zip(self.fc_layers, self.fc_activations, self.fc_dropouts)):
            x = fc(x)
            if should_store:
                activations[f'fc_{i}'] = x.detach().clone()
                
            x = act(x)
            if should_store:
                activations[f'fc_act_{i}'] = x.detach().clone()
                
            if dropout:
                x = dropout(x)
        
        # Output layer
        x = self.fc_output(x)
        if should_store:
            activations['output'] = x.detach().clone()
            self.stored_activations = activations
            return x, activations
        
        return x
    
    def get_activations(self):
        """Returns stored activations from the last forward pass"""
        return self.stored_activations