"""
Customizable ResNet model for continual learning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import get_activation, get_normalization

class BasicBlock(nn.Module):
    """Basic ResNet block with customizable activation and normalization."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, activation='relu', 
                 use_batchnorm=True, norm_after_activation=False, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.norm_after_activation = norm_after_activation
        
        # First convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not use_batchnorm)
        
        # Normalization for first conv
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else None
        
        # Activation
        self.activation = get_activation(activation)
        
        # Second convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm)
        
        # Normalization for second conv
        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else None
        
        # Downsample if needed (for shortcut connection)
        self.downsample = downsample
        
        # Activations storage
        self.stored_activations = {}
        
    def forward(self, x, store_activations=False):
        identity = x
        
        # Apply conv1
        out = self.conv1(x)
        
        # Apply norm1 if needed (before activation)
        if self.bn1 and not self.norm_after_activation:
            out = self.bn1(out)
            
        if store_activations:
            self.stored_activations['pre_activation1'] = out.detach().clone()
        
        # Apply activation
        out = self.activation(out)
        
        if store_activations:
            self.stored_activations['post_activation1'] = out.detach().clone()
        
        # Apply norm1 if needed (after activation)
        if self.bn1 and self.norm_after_activation:
            out = self.bn1(out)
            
        # Apply conv2
        out = self.conv2(out)
        
        # Apply norm2 if needed (before activation)
        if self.bn2 and not self.norm_after_activation:
            out = self.bn2(out)
            
        if store_activations:
            self.stored_activations['pre_activation2'] = out.detach().clone()
        
        # Handle shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Add identity
        out += identity
        
        # Final activation
        out = self.activation(out)
        
        if store_activations:
            self.stored_activations['post_activation2'] = out.detach().clone()
            
        # Apply norm2 if needed (after activation)
        if self.bn2 and self.norm_after_activation:
            out = self.bn2(out)
            
        return out
        
    def get_activations(self):
        return self.stored_activations


class ConfigurableResNet(nn.Module):
    """
    Configurable ResNet architecture for continual learning experiments.
    """
    def __init__(self, 
                 block=BasicBlock,
                 layers=[2, 2, 2, 2],  # ResNet18 by default
                 num_classes=10,
                 in_channels=3,
                 base_channels=64,
                 activation='relu',
                 dropout_p=0.0,
                 use_batchnorm=True,
                 norm_after_activation=False,
                 record_activations=False):
        """
        Initialize the ResNet.
        
        Parameters:
            block (nn.Module): The block type to use (BasicBlock)
            layers (list): Number of blocks in each layer
            num_classes (int): Number of output classes
            in_channels (int): Number of input channels (3 for RGB images)
            base_channels (int): Base number of channels (first layer)
            activation (str): Activation function to use
            dropout_p (float): Dropout probability before final layer
            use_batchnorm (bool): Whether to use batch normalization
            norm_after_activation (bool): Apply normalization after activation
            record_activations (bool): Whether to store activations for analysis
        """
        super(ConfigurableResNet, self).__init__()
        
        self.record_activations = record_activations
        self.in_planes = base_channels
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=not use_batchnorm)
        
        # Batch norm after first conv
        self.bn1 = nn.BatchNorm2d(base_channels) if use_batchnorm else None
        
        # Activation
        self.activation = get_activation(activation)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1, 
                                       activation=activation, use_batchnorm=use_batchnorm, 
                                       norm_after_activation=norm_after_activation)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=2, 
                                       activation=activation, use_batchnorm=use_batchnorm, 
                                       norm_after_activation=norm_after_activation)
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=2, 
                                       activation=activation, use_batchnorm=use_batchnorm,
                                       norm_after_activation=norm_after_activation)
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=2, 
                                       activation=activation, use_batchnorm=use_batchnorm,
                                       norm_after_activation=norm_after_activation)
        
        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.fc = nn.Linear(base_channels*8*block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Activation storage
        self.stored_activations = {}
                
    def _make_layer(self, block, planes, blocks, stride=1, activation='relu', 
                    use_batchnorm=True, norm_after_activation=False):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            layers = [nn.Conv2d(self.in_planes, planes * block.expansion, 
                               kernel_size=1, stride=stride, bias=not use_batchnorm)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(planes * block.expansion))
            downsample = nn.Sequential(*layers)
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, activation, 
                          use_batchnorm, norm_after_activation, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, activation, 
                               use_batchnorm, norm_after_activation))
            
        return nn.ModuleList(layers)
        
    def forward(self, x, store_activations=False):
        """
        Forward pass with optional activation storage.
        
        Parameters:
            x (torch.Tensor): Input data
            store_activations (bool): Whether to store activations
            
        Returns:
            torch.Tensor: Output logits
            dict (optional): Hidden activations if record_activations=True
        """
        # Should we store activations for this pass?
        should_store = store_activations or self.record_activations
        activations = {} if should_store else None
        
        if should_store:
            activations['input'] = x
        
        # Initial conv + norm + activation
        x = self.conv1(x)
        
        if self.bn1 and not self.norm_after_activation:
            x = self.bn1(x)
            
        if should_store:
            activations['conv1'] = x.detach().clone()
            
        x = self.activation(x)
        
        if should_store:
            activations['conv1_act'] = x.detach().clone()
            
        if self.bn1 and self.norm_after_activation:
            x = self.bn1(x)
        
        # ResNet blocks
        for layer_idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for block_idx, block in enumerate(layer):
                x = block(x, store_activations=should_store)
                if should_store:
                    # Store output of each block
                    activations[f'layer{layer_idx+1}_block{block_idx+1}'] = x.detach().clone()
                    # Store internal activations from the block if store_activations is True
                    if store_activations:
                        block_acts = block.get_activations()
                        for k, v in block_acts.items():
                            activations[f'layer{layer_idx+1}_block{block_idx+1}_{k}'] = v
        
        # Global average pooling
        x = self.avgpool(x)
        if should_store:
            activations['avgpool'] = x.detach().clone()
            
        x = torch.flatten(x, 1)
        if should_store:
            activations['flatten'] = x.detach().clone()
            
        # Dropout if specified
        if self.dropout:
            x = self.dropout(x)
            
        # Final classifier
        x = self.fc(x)
        if should_store:
            activations['output'] = x.detach().clone()
            self.stored_activations = activations
            return x, activations
            
        return x
    
    def get_activations(self):
        """Returns stored activations from the last forward pass"""
        return self.stored_activations