import torch
import torch.nn as nn
from .layers import get_activation, get_normalization

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
        for li,num_blocks in enumerate(layers):
            self._make_layer(block, base_channels*(2**li), num_blocks, stride=1 if li == 0 else 2,
                            activation=activation, use_batchnorm=use_batchnorm, 
                            norm_after_activation=norm_after_activation, 
                            layer_name=f'layer{li+1}',
                            normalization_affine=normalization_affine)
        
        self.layers['avgpool'] = nn.AdaptiveAvgPool2d((1, 1))
        self.layers['flatten'] = nn.Flatten()
        
        if dropout_p > 0:
            self.layers['dropout'] = nn.Dropout(dropout_p)
        
        self.layers['out'] = nn.Linear(base_channels*(2**(len(layers)-1))*block.expansion, num_classes)
        
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
            
        x = self.layers['out'](x)
            
        return x