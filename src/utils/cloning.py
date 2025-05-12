import torch
import torch.nn as nn
# Import the utility module and Setup the path

from ..models.layers import TransformerBatchNorm
from .monitor import NetworkMonitor
from .metrics import create_module_filter

from typing import Union, List, Dict, Set

# Replace TypeVar with Union for proper type handling
NormalizationLayer = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,  nn.GroupNorm, TransformerBatchNorm]

# Define activation function types properly
ActivationFunction = Union[nn.ReLU, nn.Sigmoid, nn.Tanh, nn.SELU, nn.GELU, nn.SiLU, nn.ELU, 
                         nn.LeakyReLU, nn.PReLU, nn.Threshold, nn.Softmax, nn.LogSoftmax, 
                         nn.Softplus, nn.Softmin, nn.Hardsigmoid, nn.Hardswish, nn.Softshrink, 
                         nn.Hardshrink, nn.Softsign, nn.GLU, nn.CELU, nn.Identity]


class CloneAwareFlatten(nn.Module):
    """
    A custom flatten module that ensures duplicated features remain adjacent when flattening
    convolutional feature maps.
    
    When cloning channels in convolutional layers, the standard nn.Flatten would arrange features
    as [a(0,0), a'(0,0), b(0,0), b'(0,0), ...] where features are grouped by spatial position.
    
    This module rearranges to keep all spatial positions of the same channel together:
    [a(0,0), a'(0,0), a(0,1), a'(0,1), ..., b(0,0), b'(0,0), ...] ensuring duplicated
    features remain adjacent.
    """
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    def forward(self, x):
        # Standard flattening for non-starting dimensions or non-4D tensors
        if x.dim() != 4 or self.start_dim > 1:
            start_dim = self.start_dim if self.start_dim >= 0 else x.dim() + self.start_dim
            end_dim = self.end_dim if self.end_dim >= 0 else x.dim() + self.end_dim
            
            shape = x.shape
            new_shape = list(shape[:start_dim]) + [-1]
            return x.reshape(*new_shape)
        
        # Special handling for 4D tensors with channel duplication
        batch_size, channels, height, width = x.shape
        
        # If channels are not even, use standard flattening
        if channels % 2 != 0:
            return x.reshape(batch_size, -1)
        
        half_channels = channels // 2
        
        # Step 1: Reshape to separate duplicated channels
        # [batch, channels, h, w] -> [batch, half_channels, 2, h, w]
        x_reshaped = x.view(batch_size, half_channels, 2, height, width)
        
        # Step 2: Permute to get the desired order
        # [batch, half_channels, 2, h, w] -> [batch, half_channels, h, w, 2]
        x_permuted = x_reshaped.permute(0, 1, 3, 4, 2)
        
        # Step 3: Flatten
        # [batch, half_channels, h, w, 2] -> [batch, half_channels * h * w * 2]
        return x_permuted.reshape(batch_size, -1)

def clone_linear(base_module: nn.Linear, cloned_module: nn.Linear):
    # Get module dimensions
    base_in_features = base_module.in_features
    base_out_features = base_module.out_features
    cloned_in_features = cloned_module.in_features
    cloned_out_features = cloned_module.out_features
    
    # Verify expansion factors are valid
    if cloned_in_features % base_in_features != 0 or cloned_out_features % base_out_features != 0:
        raise ValueError(f"Linear module dimensions are not integer multiples: "
                         f"{base_in_features}→{cloned_in_features}, {base_out_features}→{cloned_out_features}")
        
    # Calculate expansion factors
    in_expansion = cloned_in_features // base_in_features
    out_expansion = cloned_out_features // base_out_features
    
    print(f"Cloning Linear module: {base_in_features}→{cloned_in_features}, {base_out_features}→{cloned_out_features}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion] = base_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if base_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = base_module.bias.data
    return cloned_module


def clone_conv1d(base_module: nn.Conv1d, cloned_module: nn.Conv1d):
    # Get module dimensions
    base_in_channels = base_module.in_channels
    base_out_channels = base_module.out_channels
    cloned_in_channels = cloned_module.in_channels
    cloned_out_channels = cloned_module.out_channels
    # Calculate expansion factors
    in_expansion = cloned_in_channels // base_in_channels
    out_expansion = cloned_out_channels // base_out_channels
    
    print(f"Cloning Conv1d module: {base_in_channels}→{cloned_in_channels}, {base_out_channels}→{cloned_out_channels}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Verify expansion factors are valid
    if cloned_in_channels % base_in_channels != 0 or cloned_out_channels % base_out_channels != 0:
        raise ValueError(f"Conv1d module dimensions are not integer multiples: "
                         f"{base_in_channels}→{cloned_in_channels}, {base_out_channels}→{cloned_out_channels}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion, :] = base_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if base_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = base_module.bias.data
    return cloned_module

    
def clone_conv2d(base_module: nn.Conv2d, cloned_module: nn.Conv2d):
    # Get module dimensions
    base_in_channels = base_module.in_channels
    base_out_channels = base_module.out_channels
    cloned_in_channels = cloned_module.in_channels
    cloned_out_channels = cloned_module.out_channels
    # Calculate expansion factors
    in_expansion = cloned_in_channels // base_in_channels
    out_expansion = cloned_out_channels // base_out_channels
    
    print(f"Cloning Conv2d module: {base_in_channels}→{cloned_in_channels}, {base_out_channels}→{cloned_out_channels}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Verify expansion factors are valid
    if cloned_in_channels % base_in_channels != 0 or cloned_out_channels % base_out_channels != 0:
        raise ValueError(f"Conv2d module dimensions are not integer multiples: "
                         f"{base_in_channels}→{cloned_in_channels}, {base_out_channels}→{cloned_out_channels}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion, :, :] = base_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if base_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = base_module.bias.data
    return cloned_module
    

def clone_normalization(
    base_module: NormalizationLayer, 
    cloned_module: NormalizationLayer,
) -> NormalizationLayer:
    """Clone normalization layer parameters with proper handling of different types."""
    assert isinstance(cloned_module, type(base_module)), "Cloned module must be of the same type as base module"
    
    # Check properties that exist for the specific normalization type
    if hasattr(base_module, 'affine') and hasattr(cloned_module, 'affine'):
        assert base_module.affine == cloned_module.affine, "Affine property must match"
    
    # Handle BatchNorm-specific properties
    if isinstance(base_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(base_module, 'track_running_stats') and hasattr(cloned_module, 'track_running_stats'):
            assert base_module.track_running_stats == cloned_module.track_running_stats, "Track running stats property must match"
    
    # Clone weights and biases
    if hasattr(base_module, 'weight') and base_module.weight is not None and cloned_module.weight is not None:
        expansion = cloned_module.weight.data.shape[0] // base_module.weight.data.shape[0] 
        for i in range(expansion):
            cloned_module.weight.data[i::expansion] = base_module.weight.data
            if hasattr(base_module, 'bias') and base_module.bias is not None and cloned_module.bias is not None:
                cloned_module.bias.data[i::expansion] = base_module.bias.data
    
    # Clone running stats for BatchNorm layers
    if hasattr(base_module, 'running_mean') and base_module.running_mean is not None:
        if hasattr(cloned_module, 'running_mean') and cloned_module.running_mean is not None:
            expansion = cloned_module.running_mean.data.shape[0] // base_module.running_mean.data.shape[0]
            for i in range(expansion):
                cloned_module.running_mean.data[i::expansion] = base_module.running_mean.data
                cloned_module.running_var.data[i::expansion] = base_module.running_var.data
    
    # Clone num_batches_tracked for BatchNorm layers
    if hasattr(base_module, 'num_batches_tracked') and base_module.num_batches_tracked is not None:
        if hasattr(cloned_module, 'num_batches_tracked') and cloned_module.num_batches_tracked is not None:
            cloned_module.num_batches_tracked.data.copy_(base_module.num_batches_tracked.data)
    
    return cloned_module
    
    
def clone_embedding(base_module: nn.Embedding, cloned_module: nn.Embedding):
    # Get module dimensions
    base_num_embeddings = base_module.num_embeddings
    base_embedding_dim = base_module.embedding_dim
    cloned_num_embeddings = cloned_module.num_embeddings
    cloned_embedding_dim = cloned_module.embedding_dim
    
    # Calculate expansion factors
    num_expansion = cloned_num_embeddings // base_num_embeddings
    dim_expansion = cloned_embedding_dim // base_embedding_dim
    
    print(f"Cloning Embedding module: {base_num_embeddings}→{cloned_num_embeddings}, {base_embedding_dim}→{cloned_embedding_dim}, num expansion: {num_expansion}, dim expansion: {dim_expansion}")
    
    # Verify expansion factors are valid
    if cloned_num_embeddings % base_num_embeddings != 0 or cloned_embedding_dim % base_embedding_dim != 0:
        raise ValueError(f"Embedding module dimensions are not integer multiples: "
                         f"{base_num_embeddings}→{cloned_num_embeddings}, {base_embedding_dim}→{cloned_embedding_dim}")
    
    # Clone the weights with proper scaling
    for i in range(num_expansion):
        for j in range(dim_expansion):
            cloned_module.weight.data[j::dim_expansion, i::num_expansion] = base_module.weight.data 
    
    return cloned_module


def clone_activation(base_module: ActivationFunction, cloned_module: ActivationFunction) -> ActivationFunction:
    """Clone activation function parameters, handling configuration parameters properly."""
    assert isinstance(cloned_module, type(base_module)), "Cloned module must be of the same type as base module"
    
    # Handle configuration parameters for different activation types
    if isinstance(base_module, nn.LeakyReLU):
        cloned_module.negative_slope = base_module.negative_slope
    
    elif isinstance(base_module, (nn.ELU, nn.CELU)):
        cloned_module.alpha = base_module.alpha
    
    elif isinstance(base_module, nn.Threshold):
        cloned_module.threshold = base_module.threshold
        cloned_module.value = base_module.value
    
    elif isinstance(base_module, (nn.Softmax, nn.LogSoftmax)):
        cloned_module.dim = base_module.dim
    
    elif isinstance(base_module, (nn.Hardshrink, nn.Softshrink)):
        cloned_module.lambd = base_module.lambd
        
    elif isinstance(base_module, nn.GLU):
        cloned_module.dim = base_module.dim
    
    # Handle PReLU specifically (has learnable parameters)
    elif isinstance(base_module, nn.PReLU):
        if base_module.num_parameters == 1 and cloned_module.num_parameters > 1:
            # If base is a single parameter, broadcast to all channels
            cloned_module.weight.data.fill_(base_module.weight.data[0])
        elif base_module.num_parameters > 1 and cloned_module.num_parameters > 1:
            # Channel-wise parameters need proper expansion
            expansion = cloned_module.num_parameters // base_module.num_parameters
            for i in range(expansion):
                cloned_module.weight.data[i::expansion] = base_module.weight.data
        else:
            # Direct copy if dimensions match
            cloned_module.weight.data.copy_(base_module.weight.data)
    
    # Handle other parameterized activation functions if they have weights
    # This is a general catch-all for any other activation function with parameters
    elif hasattr(base_module, 'weight') and hasattr(cloned_module, 'weight'):
        if base_module.weight is not None and cloned_module.weight is not None:
            if cloned_module.weight.data.shape == base_module.weight.data.shape:
                cloned_module.weight.data.copy_(base_module.weight.data)
    
    return cloned_module


def clone_dropout(base_module: nn.Dropout, cloned_module: nn.Dropout):
    """Clone dropout module parameters."""
    assert cloned_module.p == base_module.p, "Dropout probability must match"
    # Print warning if dropout p > 0
    if cloned_module.p > 0:
        print(f"Warning: Dropout probability is set to {cloned_module.p}, cloning is not perfect")
    return cloned_module

def clone_flatten(base_module: nn.Flatten) -> CloneAwareFlatten:
    """
    Clone parameters from a standard Flatten and return a new CloneAwareFlatten.
    
    Args:
        base_module: base nn.Flatten module
        
    Returns:
        A new CloneAwareFlatten module with the same parameters
    """
    return CloneAwareFlatten(
        start_dim=base_module.start_dim,
        end_dim=base_module.end_dim
    )


def is_parameter_free(module: nn.Module) -> bool:
    """Check if a module has no parameters."""
    return len(list(module.parameters())) == 0


def clone_parameter_free(base_module: nn.Module, cloned_module: nn.Module) -> nn.Module:
    """Clone a parameter-free module."""
    assert isinstance(cloned_module, type(base_module)), "Cloned module must be of the same type as base module"
    assert is_parameter_free(base_module), "base module must be parameter free"
    assert is_parameter_free(cloned_module), "Cloned module must be parameter free"
    
    # For parameter-free modules, there's no need to copy weights
    # Just make sure they're of the same type, which we've already checked
    return cloned_module


# Validation functions

def validate_activation_cloning(base_module: ActivationFunction, cloned_module: ActivationFunction):
    assert isinstance(cloned_module, type(base_module)), "Cloned module must be of the same type as base module"
    
    # Validate configuration parameters for different activation types
    if isinstance(base_module, nn.LeakyReLU):
        assert base_module.negative_slope == cloned_module.negative_slope, "LeakyReLU negative_slope does not match"
    
    elif isinstance(base_module, (nn.ELU, nn.CELU)):
        assert base_module.alpha == cloned_module.alpha, "Alpha parameter does not match"
    
    elif isinstance(base_module, nn.Threshold):
        assert base_module.threshold == cloned_module.threshold, "Threshold value does not match"
        assert base_module.value == cloned_module.value, "Replacement value does not match"
    
    elif isinstance(base_module, (nn.Softmax, nn.LogSoftmax)):
        assert base_module.dim == cloned_module.dim, "Dimension parameter does not match"
    
    elif isinstance(base_module, (nn.Hardshrink, nn.Softshrink)):
        assert base_module.lambd == cloned_module.lambd, "Lambda parameter does not match"
        
    elif isinstance(base_module, nn.GLU):
        assert base_module.dim == cloned_module.dim, "Dimension parameter does not match"
    
    # Validate PReLU parameters
    elif isinstance(base_module, nn.PReLU):
        if base_module.num_parameters == 1 and cloned_module.num_parameters > 1:
            # All elements should be equal to the single parameter
            assert torch.all(cloned_module.weight.data == base_module.weight.data[0])
        elif base_module.num_parameters > 1 and cloned_module.num_parameters > 1:
            expansion = cloned_module.num_parameters // base_module.num_parameters
            for i in range(expansion):
                assert torch.allclose(cloned_module.weight.data[i::expansion], base_module.weight.data)
    
    print("Passed all tests")
    return True


def validate_dropout_cloning(base_module: nn.Dropout, cloned_module: nn.Dropout):
    assert cloned_module.p == base_module.p, "Dropout probability must match"
    print("Passed all tests")
    return True


def validate_embedding_cloning(base_module: nn.Embedding, cloned_module: nn.Embedding):
    num_expansion = cloned_module.num_embeddings // base_module.num_embeddings
    dim_expansion = cloned_module.embedding_dim // base_module.embedding_dim
    for j in range(num_expansion):
        for i in range(dim_expansion):
            assert torch.allclose(cloned_module.weight.data[j::num_expansion, i::dim_expansion], base_module.weight.data)
    print("Passed all tests")
    return True


def validate_normalization_cloning(base_module: NormalizationLayer, cloned_module: NormalizationLayer):
    assert isinstance(cloned_module, type(base_module)), "Cloned module must be of the same type as base module"
    
    if hasattr(base_module, 'weight') and base_module.weight is not None and hasattr(cloned_module, 'weight'):
        expansion = cloned_module.weight.data.shape[0] // base_module.weight.data.shape[0] 
        for i in range(expansion):
            assert torch.allclose(cloned_module.weight.data[i::expansion], base_module.weight.data)
            
            if hasattr(base_module, 'bias') and base_module.bias is not None and hasattr(cloned_module, 'bias'):
                assert torch.allclose(cloned_module.bias.data[i::expansion], base_module.bias.data)
    
    # Check running stats for BatchNorm layers
    if hasattr(base_module, 'running_mean') and base_module.running_mean is not None:
        if hasattr(cloned_module, 'running_mean') and cloned_module.running_mean is not None:
            expansion = cloned_module.running_mean.data.shape[0] // base_module.running_mean.data.shape[0]
            for i in range(expansion):
                assert torch.allclose(cloned_module.running_mean.data[i::expansion], base_module.running_mean.data)
                assert torch.allclose(cloned_module.running_var.data[i::expansion], base_module.running_var.data)
    
    print("Passed all tests")
    

def validate_linear_cloning(base_module: nn.Linear, cloned_module: nn.Linear):
    in_expansion = cloned_module.in_features // base_module.in_features
    out_expansion = cloned_module.out_features // base_module.out_features
    for j in range(out_expansion):
        for i in range(in_expansion):
            assert torch.allclose(cloned_module.weight.data[j::out_expansion, i::in_expansion], base_module.weight.data/in_expansion)
            assert torch.allclose(cloned_module.bias.data[j::out_expansion], base_module.bias.data)
    print("Passed all tests")
    
    
def validate_conv1d_cloning(base_module: nn.Conv1d, cloned_module: nn.Conv1d):
    in_expansion = cloned_module.in_channels // base_module.in_channels
    out_expansion = cloned_module.out_channels // base_module.out_channels
    for j in range(out_expansion):
        for i in range(in_expansion):
            assert torch.allclose(cloned_module.weight.data[j::out_expansion, i::in_expansion, :], base_module.weight.data/in_expansion)
            assert torch.allclose(cloned_module.bias.data[j::out_expansion], base_module.bias.data)
    print("Passed all tests")
    

def validate_conv2d_cloning(base_module: nn.Conv2d, cloned_module: nn.Conv2d):
    in_expansion = cloned_module.in_channels // base_module.in_channels
    out_expansion = cloned_module.out_channels // base_module.out_channels
    for j in range(out_expansion):
        for i in range(in_expansion):
            assert torch.allclose(cloned_module.weight.data[j::out_expansion, i::in_expansion, :, :], base_module.weight.data/in_expansion)
            assert torch.allclose(cloned_module.bias.data[j::out_expansion], base_module.bias.data)
    print("Passed all tests")



def clone_module(
    base_module: nn.Module, 
    cloned_module: nn.Module,
) -> bool:
    """
    Clone parameters from a base module to a cloned module.
    
    Args:
        base_module: base module with smaller dimensions
        cloned_module: Target module with larger dimensions
        
    Returns:
        bool: True if cloning was successful, False otherwise
    """
    success = True
    
    # Define normalization and activation types inline for easier checking
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
    activation_types = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.SELU, nn.GELU, nn.SiLU, nn.ELU, nn.LeakyReLU, 
                       nn.PReLU, nn.Threshold, nn.Softmax, nn.LogSoftmax, nn.Softplus, nn.Softmin, 
                       nn.Hardsigmoid, nn.Hardswish, nn.Softshrink, nn.Hardshrink, nn.Softsign, 
                       nn.GLU, nn.CELU, nn.Identity)
    
    if isinstance(base_module, nn.Linear):
        clone_linear(base_module, cloned_module)
    elif isinstance(base_module, nn.Conv1d):
        clone_conv1d(base_module, cloned_module)
    elif isinstance(base_module, nn.Conv2d):
        clone_conv2d(base_module, cloned_module)
    elif isinstance(base_module, norm_types):
        clone_normalization(base_module, cloned_module)
    elif isinstance(base_module, nn.Embedding):
        clone_embedding(base_module, cloned_module)
    elif isinstance(base_module, activation_types):
        clone_activation(base_module, cloned_module)
    elif isinstance(base_module, nn.Dropout):
        clone_dropout(base_module, cloned_module)
    elif isinstance(base_module, nn.Flatten):
        pass # Flatten is handled separately
    elif is_parameter_free(base_module) and is_parameter_free(cloned_module):
        clone_parameter_free(base_module, cloned_module)
    else:
        success = False
        print(f"Unsupported module type: {type(base_module)}")
    
    return success



def model_clone(base_model: nn.Module, cloned_model: nn.Module) -> nn.Module:
    """
    Clone parameters from a base model to a cloned model.
    
    Args:
        base_model: base model with smaller dimensions
        cloned_model: Target model with larger dimensions
        
    Returns:
        cloned_model: The target model with cloned parameters
    """
    # First, replace all Flatten modules with CloneAwareFlatten
    for name, module in list(cloned_model.named_modules()):
        if isinstance(module, nn.Flatten):
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            
            # Find parent module to modify
            if parent_name:
                parent = cloned_model.get_submodule(parent_name)
            else:
                parent = cloned_model
                
            # Create and replace with CloneAwareFlatten
            setattr(parent, module_name, CloneAwareFlatten(
                start_dim=module.start_dim,
                end_dim=module.end_dim
            ))
            print(f"Replaced Flatten with CloneAwareFlatten at {name}")
    
    # Second, handle direct parameters of the model (not within modules)
    for name, param in list(base_model.named_parameters(recurse=False)):
        if hasattr(cloned_model, name):
            base_param = getattr(base_model, name)
            cloned_param = getattr(cloned_model, name)
            
            # Check if dimensions differ and can be expanded
            if base_param.shape != cloned_param.shape:
                # For embedding dimensions (typically last dimension in transformers)
                if len(base_param.shape) >= 2 and base_param.shape[:-1] == cloned_param.shape[:-1]:
                    base_dim = base_param.shape[-1]
                    cloned_dim = cloned_param.shape[-1]
                    
                    if cloned_dim % base_dim == 0:
                        expansion = cloned_dim // base_dim
                        # Duplicate across embedding dimension
                        for i in range(expansion):
                            cloned_param.data[..., i::expansion] = base_param.data
                        print(f"Cloned parameter {name} with embedding expansion {expansion}")
                    else:
                        print(f"Warning: Parameter {name} dimensions don't match and can't be expanded automatically")
                else:
                    print(f"Warning: Parameter {name} shapes don't match and can't be expanded automatically")
            else:
                # Exact shape match, just copy
                cloned_param.data.copy_(base_param.data)
                print(f"Cloned parameter {name} with direct copy")
    
    # Finally, process module parameters
    for name, base_module in base_model.named_modules():
        try:
            cloned_module = cloned_model.get_submodule(name)
            print(f"Cloning module {name}")
            clone_module(base_module, cloned_module)
        except AttributeError:
            print(f"Warning: Could not find matching module for {name}")
    
    return cloned_model



def test_activation_cloning(base_model, cloned_model, input, target, model_name=None, tolerance=1e-3, check_equality=False, eps = 1e-10):
    criterion = nn.CrossEntropyLoss()
    filter = create_module_filter(['default'], model_name,)
    base_monitor = NetworkMonitor(base_model,filter)
    cloned_monitor = NetworkMonitor(cloned_model,filter)
    base_monitor.register_hooks()
    cloned_monitor.register_hooks()

    y1 = base_model(input)
    y2 = cloned_model(input)
    l1 = criterion(y1, target)
    l2 = criterion(y2, target)
    l1.backward()
    l2.backward()
    cloned_monitor.remove_hooks()
    base_monitor.remove_hooks()
    success = True 
    
    if check_equality and  torch.allclose(y1, y2,atol=tolerance):
        print("Outputs do not match after cloning")
        success = False 

    cloning_r2 = dict()
    for act_type in ['forward', 'backward']:
        if act_type == 'forward':
            base_acts = base_monitor.get_latest_activations()
            clone_acts = cloned_monitor.get_latest_activations()
        elif act_type == 'backward':
            base_acts = base_monitor.get_latest_gradients()
            clone_acts = cloned_monitor.get_latest_gradients()

        for key, a1 in base_acts.items():
            a2 = clone_acts[key]
            s1, s2 = torch.tensor(a1.shape), torch.tensor(a2.shape)
            print(f"key: {key}, a1: {a1.shape}, a2: {a2.shape}")
            i = (s1 != s2).nonzero()
            if len(i)==0:
                if check_equality:
                    assert torch.allclose(a1, a2, atol=tolerance), f"Activations for {key} do not match"
            elif len(i)==1:
                i = i[0][0]
                expansion = a2.shape[i] // a1.shape[i]
                # check expansion depending on the dimension 
                slices = []
                for j in range(expansion):
                    print(f"mismatch dim: {i}, checking slice: {j}, expansion: {expansion}")
                    if i==0:
                        slice = a2[j::expansion]
                    elif i==1:
                        slice = a2[:, j::expansion]
                    elif i==2:
                        slice = a2[:, :, j::expansion]
                    elif i==3:
                        slice = a2[:, :, :, j::expansion]
                    slices.append(slice)
                    if check_equality and not torch.allclose(slice, a1, atol=tolerance):
                        print(f"Activations for {key} do not match")
                        success = False 
                slices = torch.stack(slices)
                if slices.shape[0]>1:
                    print(f"slices for {key} shape = {slices.shape}")
                    var, var_all = slices.std(dim=0), a2.std(dim=i, keepdim=True)
                    # cloning r2 = 1 - un explained variance
                    r2 = 1 - ((var)/(var_all+eps)).mean().item()
                    print(f"r2 score {key} {act_type} activations is {r2}")
                    cloning_r2[f'{key}_{act_type}'] = r2
                    if r2<1-tolerance:
                        print(f"r2 is lower than the threshold: 1-{tolerance}")
                        success = False 

            elif len(i)>1:
                assert False, f"Activations for {key} more than one dimension mismatch, this is unexpected behavior"
                    
        print(f"All {act_type} activations match after cloning up to tolerance {tolerance}")
    overall_r2 = sum([r2 for key, r2 in cloning_r2.items() if 'forward' in key])/len(cloning_r2)
    cloning_r2['mean'] = overall_r2
    return success, cloning_r2
    
def expand_model(current_model, cfg, expansion_factor):
    """
    Create an expanded model by cloning the current model with an expansion factor.
    
    Args:
        current_model: The base model to clone
        cfg: Configuration object containing model specifications
        expansion_factor: Factor by which to expand model capacity
        
    Returns:
        An initialized expanded model with cloned parameters
    """
    from ..models import MLP, CNN, ResNet, VisionTransformer
    from omegaconf import OmegaConf
    import copy
    
    # Get the model type from config
    model_name = cfg.model.name.lower()
    
    # Convert model config to a mutable dictionary
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    
    # Ensure dataset parameters are added to model params
    
    # Remove name and _target_ from parameters
    if 'name' in model_params:
        del model_params['name']
    if '_target_' in model_params:
        del model_params['_target_']
    
    # Expand appropriate dimensions based on model type
    if model_name == 'mlp':
        # Add input_size and output_size for MLP
        model_params['input_size'] = cfg.dataset.input_size
        model_params['output_size'] = cfg.dataset.num_classes
        
        # Expand hidden sizes
        model_params['hidden_sizes'] = [size * expansion_factor for size in model_params['hidden_sizes']]
        expanded_model = MLP(**model_params)
        
    elif model_name == 'cnn':
        # Add CNN specific parameters
        model_params['num_classes'] = cfg.dataset.num_classes
        model_params['in_channels'] = cfg.dataset.in_channels
        model_params['input_size'] = cfg.dataset.img_size
        
        # Expand conv channels and fc hidden units
        model_params['conv_channels'] = [channels * expansion_factor for channels in model_params['conv_channels']]
        model_params['fc_hidden_units'] = [units * expansion_factor for units in model_params['fc_hidden_units']]
        expanded_model = CNN(**model_params)
        
    elif model_name == 'resnet':
        model_params['num_classes'] = cfg.dataset.num_classes
        # Add ResNet specific parameters
        model_params['in_channels'] = cfg.dataset.in_channels
        
        # Expand base channels
        model_params['base_channels'] = model_params['base_channels'] * expansion_factor
        expanded_model = ResNet(**model_params)
        
    elif model_name == 'vit':
        # Add ViT specific parameters
        model_params['num_classes'] = cfg.dataset.num_classes
        model_params['img_size'] = cfg.dataset.img_size
        model_params['in_channels'] = cfg.dataset.in_channels
        
        # Expand embed dimensions
        model_params['embed_dim'] = model_params['embed_dim'] * expansion_factor
        expanded_model = VisionTransformer(**model_params)
        
    else:
        raise ValueError(f"Unsupported model type for cloning: {model_name}")
    
    return expanded_model


def create_cloned_model(current_model, cfg, expansion_factor):
    # Clone parameters from the current model to the expanded model
    expanded_model = expand_model(current_model, cfg, expansion_factor)
    expanded_model = model_clone(current_model, expanded_model)
    return expanded_model
    
    


def test_various_models_cloning(normalization='none', drpout_p=0.0, activation='relu', tolerance=1e-3,check_equality=False):
    from src.models import MLP, CNN, ResNet, VisionTransformer
    
    # generate random input and targets
    x_flat = torch.randn(32, 10) # for MLP 
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 2, (32,))
    
    
    base_model = MLP(input_size=10, output_size=2, hidden_sizes=[64, 32,], activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = MLP(input_size=10, output_size=2, hidden_sizes=[64*2, 32*2,], activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = model_clone(base_model, cloned_model)
    success, _ = test_activation_cloning(base_model, cloned_model, x_flat, y, tolerance=tolerance, check_equality=check_equality, model_name='mlp')
    print(f">>>> MLP cloning success: {success}")
    
    base_model = CNN(in_channels=3, num_classes=2, conv_channels=[64, 128, 256], activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = CNN(in_channels=3, num_classes=2, conv_channels=[64*2, 128*2, 256*2], activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = model_clone(base_model, cloned_model)
    success, _  = test_activation_cloning(base_model, cloned_model, x, y, tolerance=tolerance, check_equality=check_equality, model_name='cnn')
    print(f">>>> CNN cloning success: {success}")

    base_model = ResNet(in_channels=3, num_classes=2, base_channels=64, activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = ResNet(in_channels=3, num_classes=2, base_channels=64*2, activation=activation, dropout_p=drpout_p, normalization=normalization)
    cloned_model = model_clone(base_model, cloned_model)
    success, _  = test_activation_cloning(base_model, cloned_model, x, y, tolerance=tolerance, check_equality=check_equality, model_name='resnet')
    print(f">>>> ResNet cloning success: {success}")
    
    
    base_model = VisionTransformer(
        in_channels=3, 
        num_classes=2, 
        embed_dim=64, 
        depth=2, 
        dropout_p=drpout_p,
        attn_drop_rate=drpout_p,
        activation=activation,)

    cloned_model = VisionTransformer(
        in_channels=3, 
        num_classes=2, 
        patch_size=4, 
        embed_dim=64*2, 
        depth=2, 
        dropout_p=drpout_p,
        attn_drop_rate=drpout_p,
        activation=activation,)

    cloned_model = model_clone(base_model, cloned_model)

    success, _  =  test_activation_cloning(base_model, cloned_model, x, y, tolerance=tolerance, check_equality=check_equality, model_name='vit')
    print(f">>>> ViT cloning success: {success}")
    
# if __name__ == "__main__":
#     # Test the cloning functionality with various models
#     for activation in ['relu', 'tanh', 'gelu']:
#         for normalization in ['none', 'layer', 'batch']:
#             test_various_models_cloning(activation=activation, normalization=normalization,drpout_p=0.1, tolerance=1e-8, check_equality=False)    
# if __name__ == "__main__":
#     test_various_models_cloning(activation=activation, normalization=normalization,drpout_p=0.0, tolerance=0.1)
    