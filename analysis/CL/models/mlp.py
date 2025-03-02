"""
Customizable MLP model for continual learning experiments.
"""

import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

class CustomizableMLP(nn.Module):
    def __init__(self, 
                 input_size=784, 
                 hidden_sizes=[512, 256, 128], 
                 output_size=10, 
                 activation='relu',
                 dropout_p=0.0,
                 normalization=None,
                 norm_after_activation=False,
                 bias=True,
                 record_activations=False):
        """
        Fully customizable MLP that supports various activations and normalizations.
        
        Parameters:
            input_size (int): Dimensionality of input features
            hidden_sizes (list): List of hidden layer dimensions
            output_size (int): Number of output classes
            activation (str): Activation function to use ('relu', 'tanh', 'sigmoid', etc.)
            dropout_p (float): Dropout probability (0 to disable)
            normalization (str): Normalization to use ('batch', 'layer', None)
            norm_after_activation (bool): If True, apply normalization after activation
            bias (bool): Whether to include bias terms in linear layers
            record_activations (bool): Whether to store activations for analysis
        """
        super(CustomizableMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.record_activations = record_activations
        
        # Build network
        layers = []
        in_features = input_size
        
        # Store module references for easier access
        self.linear_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for hidden_size in hidden_sizes:
            # Linear layer
            linear = nn.Linear(in_features, hidden_size, bias=bias)
            layers.append(linear)
            self.linear_layers.append(linear)
            
            # Activation
            act_layer = get_activation(activation)
            self.activation_layers.append(act_layer)
            
            # Normalization and order
            norm_layer = get_normalization(normalization, hidden_size) if normalization else None
            self.norm_layers.append(norm_layer)
            
            if norm_after_activation:
                layers.append(act_layer)
                if norm_layer:
                    layers.append(norm_layer)
            else:
                if norm_layer:
                    layers.append(norm_layer)
                layers.append(act_layer)
            
            # Dropout
            if dropout_p > 0:
                dropout = nn.Dropout(dropout_p)
                layers.append(dropout)
                self.dropout_layers.append(dropout)
            else:
                self.dropout_layers.append(None)
            
            in_features = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(in_features, output_size, bias=bias)
        layers.append(self.output_layer)
        
        # Sequential container
        self.layers = nn.Sequential(*layers)
        
        # Activation storage
        self.stored_activations = {}
        
    def forward(self, x, store_activations=False):
        """
        Forward pass with optional activation storage.
        
        Parameters:
            x (torch.Tensor): Input data with shape [batch_size, input_size]
            store_activations (bool): Whether to store activations from this pass
        
        Returns:
            torch.Tensor: Output logits
            dict (optional): Hidden activations if record_activations=True
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Should we store activations for this pass?
        should_store = store_activations or self.record_activations
        activations = {} if should_store else None
            
        if should_store:
            activations['input'] = x
        
        # Apply all layers except the output layer
        h = x
        for idx, layer in enumerate(self.layers[:-1]):  # exclude output layer
            h = layer(h)
            # Store activations at meaningful points (after each block)
            if should_store and isinstance(layer, nn.Linear):
                activations[f'linear_{idx}'] = h.detach().clone()
                
            # After full block (activation & normalization)
            if should_store and (
                isinstance(layer, nn.ReLU) or 
                isinstance(layer, nn.Tanh) or 
                isinstance(layer, nn.Sigmoid)):
                activations[f'activation_{idx}'] = h.detach().clone()
        
        # Output layer
        output = self.layers[-1](h)
        
        if should_store:
            activations['output'] = output
            self.stored_activations = activations
            return output, activations
        
        return output
    
    def get_activations(self):
        """Returns stored activations from the last forward pass"""
        return self.stored_activations