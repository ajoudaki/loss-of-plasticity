import torch
import torch.nn as nn
from collections import defaultdict
from typing import Union, List, Dict, Set, Callable, Optional

# Define type hints
NormalizationLayer = Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm]
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


def clone_linear(src_module: nn.Linear, cloned_module: nn.Linear):
    """Clone parameters from a source linear module to a cloned module."""
    # Get module dimensions
    src_in_features = src_module.in_features
    src_out_features = src_module.out_features
    cloned_in_features = cloned_module.in_features
    cloned_out_features = cloned_module.out_features
    
    # Verify expansion factors are valid
    if cloned_in_features % src_in_features != 0 or cloned_out_features % src_out_features != 0:
        raise ValueError(f"Linear module dimensions are not integer multiples: "
                     f"{src_in_features}→{cloned_in_features}, {src_out_features}→{cloned_out_features}")
        
    # Calculate expansion factors
    in_expansion = cloned_in_features // src_in_features
    out_expansion = cloned_out_features // src_out_features
    
    print(f"Cloning Linear module: {src_in_features}→{cloned_in_features}, {src_out_features}→{cloned_out_features}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion] = src_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if src_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = src_module.bias.data
    return cloned_module


def clone_conv1d(src_module: nn.Conv1d, cloned_module: nn.Conv1d):
    """Clone parameters from a source 1D conv module to a cloned module."""
    # Get module dimensions
    src_in_channels = src_module.in_channels
    src_out_channels = src_module.out_channels
    cloned_in_channels = cloned_module.in_channels
    cloned_out_channels = cloned_module.out_channels
    # Calculate expansion factors
    in_expansion = cloned_in_channels // src_in_channels
    out_expansion = cloned_out_channels // src_out_channels
    
    print(f"Cloning Conv1d module: {src_in_channels}→{cloned_in_channels}, {src_out_channels}→{cloned_out_channels}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Verify expansion factors are valid
    if cloned_in_channels % src_in_channels != 0 or cloned_out_channels % src_out_channels != 0:
        raise ValueError(f"Conv1d module dimensions are not integer multiples: "
                     f"{src_in_channels}→{cloned_in_channels}, {src_out_channels}→{cloned_out_channels}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion, :] = src_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if src_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = src_module.bias.data
    return cloned_module

    
def clone_conv2d(src_module: nn.Conv2d, cloned_module: nn.Conv2d):
    """Clone parameters from a source 2D conv module to a cloned module."""
    # Get module dimensions
    src_in_channels = src_module.in_channels
    src_out_channels = src_module.out_channels
    cloned_in_channels = cloned_module.in_channels
    cloned_out_channels = cloned_module.out_channels
    # Calculate expansion factors
    in_expansion = cloned_in_channels // src_in_channels
    out_expansion = cloned_out_channels // src_out_channels
    
    print(f"Cloning Conv2d module: {src_in_channels}→{cloned_in_channels}, {src_out_channels}→{cloned_out_channels}, in expansion: {in_expansion}, out expansion: {out_expansion}")
    
    # Verify expansion factors are valid
    if cloned_in_channels % src_in_channels != 0 or cloned_out_channels % src_out_channels != 0:
        raise ValueError(f"Conv2d module dimensions are not integer multiples: "
                     f"{src_in_channels}→{cloned_in_channels}, {src_out_channels}→{cloned_out_channels}")
    
    # Clone the weights with proper scaling
    for i in range(in_expansion):
        for j in range(out_expansion):
            cloned_module.weight.data[j::out_expansion, i::in_expansion, :, :] = src_module.weight.data / in_expansion
    
    # Clone the bias if present (no scaling needed for bias)
    if src_module.bias is not None and cloned_module.bias is not None:
        for j in range(out_expansion):
            cloned_module.bias.data[j::out_expansion] = src_module.bias.data
    return cloned_module
    

def clone_normalization(
    src_module: NormalizationLayer, 
    cloned_module: NormalizationLayer,
) -> NormalizationLayer:
    """Clone normalization layer parameters with proper handling of different types."""
    assert isinstance(cloned_module, type(src_module)), "Cloned module must be of the same type as source module"
    
    # Check properties that exist for the specific normalization type
    if hasattr(src_module, 'affine') and hasattr(cloned_module, 'affine'):
        assert src_module.affine == cloned_module.affine, "Affine property must match"
    
    # Handle BatchNorm-specific properties
    if isinstance(src_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if hasattr(src_module, 'track_running_stats') and hasattr(cloned_module, 'track_running_stats'):
            assert src_module.track_running_stats == cloned_module.track_running_stats, "Track running stats property must match"
    
    # Clone weights and biases
    if hasattr(src_module, 'weight') and src_module.weight is not None and cloned_module.weight is not None:
        expansion = cloned_module.weight.data.shape[0] // src_module.weight.data.shape[0] 
        for i in range(expansion):
            cloned_module.weight.data[i::expansion] = src_module.weight.data
            if hasattr(src_module, 'bias') and src_module.bias is not None and cloned_module.bias is not None:
                cloned_module.bias.data[i::expansion] = src_module.bias.data
    
    # Clone running stats for BatchNorm layers
    if hasattr(src_module, 'running_mean') and src_module.running_mean is not None:
        if hasattr(cloned_module, 'running_mean') and cloned_module.running_mean is not None:
            expansion = cloned_module.running_mean.data.shape[0] // src_module.running_mean.data.shape[0]
            for i in range(expansion):
                cloned_module.running_mean.data[i::expansion] = src_module.running_mean.data
                cloned_module.running_var.data[i::expansion] = src_module.running_var.data
    
    # Clone num_batches_tracked for BatchNorm layers
    if hasattr(src_module, 'num_batches_tracked') and src_module.num_batches_tracked is not None:
        if hasattr(cloned_module, 'num_batches_tracked') and cloned_module.num_batches_tracked is not None:
            cloned_module.num_batches_tracked.data.copy_(src_module.num_batches_tracked.data)
    
    return cloned_module
    
    
def clone_embedding(src_module: nn.Embedding, cloned_module: nn.Embedding):
    """Clone parameters from a source embedding module to a cloned module."""
    # Get module dimensions
    src_num_embeddings = src_module.num_embeddings
    src_embedding_dim = src_module.embedding_dim
    cloned_num_embeddings = cloned_module.num_embeddings
    cloned_embedding_dim = cloned_module.embedding_dim
    
    # Calculate expansion factors
    num_expansion = cloned_num_embeddings // src_num_embeddings
    dim_expansion = cloned_embedding_dim // src_embedding_dim
    
    print(f"Cloning Embedding module: {src_num_embeddings}→{cloned_num_embeddings}, {src_embedding_dim}→{cloned_embedding_dim}, num expansion: {num_expansion}, dim expansion: {dim_expansion}")
    
    # Verify expansion factors are valid
    if cloned_num_embeddings % src_num_embeddings != 0 or cloned_embedding_dim % src_embedding_dim != 0:
        raise ValueError(f"Embedding module dimensions are not integer multiples: "
                     f"{src_num_embeddings}→{cloned_num_embeddings}, {src_embedding_dim}→{cloned_embedding_dim}")
    
    # Clone the weights with proper scaling
    for i in range(num_expansion):
        for j in range(dim_expansion):
            cloned_module.weight.data[j::dim_expansion, i::num_expansion] = src_module.weight.data 
    
    return cloned_module


def clone_activation(src_module: ActivationFunction, cloned_module: ActivationFunction) -> ActivationFunction:
    """Clone activation function parameters, handling configuration parameters properly."""
    assert isinstance(cloned_module, type(src_module)), "Cloned module must be of the same type as source module"
    
    # Handle configuration parameters for different activation types
    if isinstance(src_module, nn.LeakyReLU):
        cloned_module.negative_slope = src_module.negative_slope
    
    elif isinstance(src_module, (nn.ELU, nn.CELU)):
        cloned_module.alpha = src_module.alpha
    
    elif isinstance(src_module, nn.Threshold):
        cloned_module.threshold = src_module.threshold
        cloned_module.value = src_module.value
    
    elif isinstance(src_module, (nn.Softmax, nn.LogSoftmax)):
        cloned_module.dim = src_module.dim
    
    elif isinstance(src_module, (nn.Hardshrink, nn.Softshrink)):
        cloned_module.lambd = src_module.lambd
        
    elif isinstance(src_module, nn.GLU):
        cloned_module.dim = src_module.dim
    
    # Handle PReLU specifically (has learnable parameters)
    elif isinstance(src_module, nn.PReLU):
        if src_module.num_parameters == 1 and cloned_module.num_parameters > 1:
            # If source is a single parameter, broadcast to all channels
            cloned_module.weight.data.fill_(src_module.weight.data[0])
        elif src_module.num_parameters > 1 and cloned_module.num_parameters > 1:
            # Channel-wise parameters need proper expansion
            expansion = cloned_module.num_parameters // src_module.num_parameters
            for i in range(expansion):
                cloned_module.weight.data[i::expansion] = src_module.weight.data
        else:
            # Direct copy if dimensions match
            cloned_module.weight.data.copy_(src_module.weight.data)
    
    # Handle other parameterized activation functions if they have weights
    # This is a general catch-all for any other activation function with parameters
    elif hasattr(src_module, 'weight') and hasattr(cloned_module, 'weight'):
        if src_module.weight is not None and cloned_module.weight is not None:
            if cloned_module.weight.data.shape == src_module.weight.data.shape:
                cloned_module.weight.data.copy_(src_module.weight.data)
    
    return cloned_module


def clone_dropout(src_module: nn.Dropout, cloned_module: nn.Dropout):
    """Clone dropout module parameters."""
    assert cloned_module.p == src_module.p, "Dropout probability must match"
    # Print warning if dropout p > 0
    if cloned_module.p > 0:
        print(f"Warning: Dropout probability is set to {cloned_module.p}, cloning is not perfect")
    return cloned_module


def clone_flatten(src_module: nn.Flatten) -> CloneAwareFlatten:
    """
    Clone parameters from a standard Flatten and return a new CloneAwareFlatten.
    
    Args:
        src_module: Source nn.Flatten module
        
    Returns:
        A new CloneAwareFlatten module with the same parameters
    """
    return CloneAwareFlatten(
        start_dim=src_module.start_dim,
        end_dim=src_module.end_dim
    )


def is_parameter_free(module: nn.Module) -> bool:
    """Check if a module has no parameters."""
    return len(list(module.parameters())) == 0


def clone_parameter_free(src_module: nn.Module, cloned_module: nn.Module) -> nn.Module:
    """Clone a parameter-free module."""
    assert isinstance(cloned_module, type(src_module)), "Cloned module must be of the same type as source module"
    assert is_parameter_free(src_module), "Source module must be parameter free"
    assert is_parameter_free(cloned_module), "Cloned module must be parameter free"
    
    # For parameter-free modules, there's no need to copy weights
    # Just make sure they're of the same type, which we've already checked
    return cloned_module


def clone_module(
    src_module: nn.Module, 
    cloned_module: nn.Module,
) -> bool:
    """
    Clone parameters from a source module to a cloned module.
    
    Args:
        src_module: Source module with smaller dimensions
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
    
    if isinstance(src_module, nn.Linear):
        clone_linear(src_module, cloned_module)
    elif isinstance(src_module, nn.Conv1d):
        clone_conv1d(src_module, cloned_module)
    elif isinstance(src_module, nn.Conv2d):
        clone_conv2d(src_module, cloned_module)
    elif isinstance(src_module, norm_types):
        clone_normalization(src_module, cloned_module)
    elif isinstance(src_module, nn.Embedding):
        clone_embedding(src_module, cloned_module)
    elif isinstance(src_module, activation_types):
        clone_activation(src_module, cloned_module)
    elif isinstance(src_module, nn.Dropout):
        clone_dropout(src_module, cloned_module)
    elif isinstance(src_module, nn.Flatten):
        pass # Flatten is handled separately
    elif is_parameter_free(src_module) and is_parameter_free(cloned_module):
        clone_parameter_free(src_module, cloned_module)
    else:
        success = False
        print(f"Unsupported module type: {type(src_module)}")
    
    return success


def clone_model(src_model: nn.Module, cloned_model: nn.Module) -> nn.Module:
    """
    Clone parameters from a source model to a cloned model.
    
    Args:
        src_model: Source model with smaller dimensions
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
    for name, param in list(src_model.named_parameters(recurse=False)):
        if hasattr(cloned_model, name):
            src_param = getattr(src_model, name)
            cloned_param = getattr(cloned_model, name)
            
            # Check if dimensions differ and can be expanded
            if src_param.shape != cloned_param.shape:
                # For embedding dimensions (typically last dimension in transformers)
                if len(src_param.shape) >= 2 and src_param.shape[:-1] == cloned_param.shape[:-1]:
                    src_dim = src_param.shape[-1]
                    cloned_dim = cloned_param.shape[-1]
                    
                    if cloned_dim % src_dim == 0:
                        expansion = cloned_dim // src_dim
                        # Duplicate across embedding dimension
                        for i in range(expansion):
                            cloned_param.data[..., i::expansion] = src_param.data
                        print(f"Cloned parameter {name} with embedding expansion {expansion}")
                    else:
                        print(f"Warning: Parameter {name} dimensions don't match and can't be expanded automatically")
                else:
                    print(f"Warning: Parameter {name} shapes don't match and can't be expanded automatically")
            else:
                # Exact shape match, just copy
                cloned_param.data.copy_(src_param.data)
                print(f"Cloned parameter {name} with direct copy")
    
    # Finally, process module parameters
    for name, src_module in src_model.named_modules():
        try:
            if name == "":
                continue  # Skip the root module
            cloned_module = cloned_model.get_submodule(name)
            print(f"Cloning module {name}")
            clone_module(src_module, cloned_module)
        except AttributeError:
            print(f"Warning: Could not find matching module for {name}")
    
    return cloned_model


def test_activation_cloning(src_model, cloned_model, input_data, tolerance=1e-3):
    """Test if activations match between source and cloned models."""
    from src.utils.monitor import NetworkMonitor
    
    src_monitor = NetworkMonitor(src_model)
    cloned_monitor = NetworkMonitor(cloned_model)
    src_monitor.register_hooks()
    cloned_monitor.register_hooks()

    with torch.no_grad():
        y1 = src_model(input_data)
        y2 = cloned_model(input_data)
    
    assert torch.allclose(y1, y2, atol=tolerance), "Outputs do not match after cloning"

    acts1 = src_monitor.get_latest_activations()
    acts2 = cloned_monitor.get_latest_activations()

    for key, a1 in acts1.items():
        if key not in acts2:
            continue
        a2 = acts2[key]
        s1, s2 = torch.tensor(a1.shape), torch.tensor(a2.shape)
        i = (s1 != s2).nonzero()
        if len(i) == 0:
            assert torch.allclose(a1, a2, atol=tolerance), f"Activations for {key} do not match"
        if len(i) == 1:
            expansion = a2.shape[i[0][0]] // a1.shape[i[0][0]]
            for j in range(expansion):
                assert torch.allclose(a2[:, j::expansion], a1, atol=tolerance), f"Activations for {key} do not match"
        if len(i) > 1:
            assert False, f"Activations for {key} have more than one dimension mismatch, this is unexpected behavior"
            
    print(f"All activations match after cloning up to tolerance {tolerance}")
    
    # Clean up monitors
    src_monitor.remove_hooks()
    cloned_monitor.remove_hooks()


def create_cloned_model(src_model, full_config, clone_factor):
    """Create a cloned model with scaled dimensions based on the source model."""
    from src.models.model_factory import create_model
    from copy import deepcopy
    import omegaconf
    
    # Create a deep copy of the full config
    cloned_config = omegaconf.OmegaConf.create(
        omegaconf.OmegaConf.to_container(full_config, resolve=True)
    )
    
    model_config = full_config.model
    
    # Scale hidden dimensions based on model type
    if model_config.name.lower() == 'mlp':
        cloned_config.model.hidden_sizes = [size * clone_factor for size in model_config.hidden_sizes]
    
    elif model_config.name.lower() == 'cnn':
        cloned_config.model.conv_channels = [size * clone_factor for size in model_config.conv_channels]
        
    elif model_config.name.lower() == 'resnet':
        cloned_config.model.base_channels = model_config.base_channels * clone_factor
        
    elif model_config.name.lower() == 'vit':
        cloned_config.model.embed_dim = model_config.embed_dim * clone_factor
        if 'mlp_ratio' in cloned_config.model:
            # Keep the same expansion ratio for MLP blocks
            pass
    
    # Create the cloned model with scaled dimensions
    cloned_model = create_model(cloned_config)
    
    # Clone parameters from source to cloned model
    cloned_model = clone_model(src_model, cloned_model)
    
    return cloned_model