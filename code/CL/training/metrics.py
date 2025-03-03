"""
Metrics for neural network analysis in continual learning.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import scipy


def compute_accuracy(model, data_loader, device=None):
    """
    Compute accuracy on a dataset.
    
    Parameters:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): Data loader
        device (torch.device, optional): Device to use
        
    Returns:
        float: Accuracy
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle case when output is a tuple (e.g., outputs and activations)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get just the model predictions
                
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    return correct / total


def compute_forgetting(old_acc, new_acc):
    """
    Compute forgetting between two accuracy measurements.
    
    Parameters:
        old_acc (float): Previous accuracy
        new_acc (float): Current accuracy
        
    Returns:
        float: Forgetting (positive if performance degraded)
    """
    return max(0, old_acc - new_acc)


def compute_effective_rank(matrix, eps=1e-12):
    """
    Compute the effective rank of a matrix using SVD.
    
    Parameters:
        matrix (torch.Tensor): Input matrix
        eps (float): Small constant for numerical stability
        
    Returns:
        float: Effective rank
    """
    if len(matrix.shape) > 2:
        # Reshape to 2D if needed
        matrix = matrix.reshape(matrix.shape[0], -1)
    
    # Use SVD to compute singular values
    try:
        U, S, V = torch.svd(matrix.float())
    except:
        # Fallback to scipy for numerical stability
        S = torch.tensor(scipy.linalg.svdvals(matrix.cpu().numpy()))
    
    # Normalize singular values to get "probabilities"
    S_norm = S / (S.sum() + eps)
    
    # Compute entropy of these "probabilities"
    entropy = -torch.sum(S_norm * torch.log(S_norm + eps))
    
    # Effective rank is exp(entropy)
    return torch.exp(entropy).item()


def compute_rank_metrics(model, layers=None):
    """
    Compute rank-related metrics for model weights.
    
    Parameters:
        model (nn.Module): The model to analyze
        layers (list, optional): List of layer names to analyze. If None, analyze all linear layers.
        
    Returns:
        dict: Dictionary of rank metrics by layer
    """
    metrics = {}
    
    # If no layers specified, find all linear layers
    if layers is None:
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                layers.append(name)
    
    # Compute metrics for each layer
    for name, module in model.named_modules():
        if name in layers or name.endswith('.weight'):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                
                # Skip if weight is None
                if weight is None:
                    continue
                
                # Compute SVD and rank
                try:
                    if isinstance(module, nn.Conv2d):
                        # Reshape conv weights for rank computation
                        w_flat = weight.view(weight.size(0), -1)
                    else:
                        w_flat = weight
                    
                    # Compute effective rank
                    eff_rank = compute_effective_rank(w_flat)
                    
                    # Compute additional metrics
                    metrics[f"{name}_eff_rank"] = eff_rank
                    metrics[f"{name}_weight_norm"] = torch.norm(weight).item()
                    metrics[f"{name}_weight_mean"] = weight.mean().item()
                    metrics[f"{name}_weight_std"] = weight.std().item()
                    
                    # Count near-zero weights (weight sparsity)
                    near_zero = (weight.abs() < 1e-4).float().mean().item()
                    metrics[f"{name}_weight_sparsity"] = near_zero
                    
                except Exception as e:
                    print(f"Error computing rank metrics for {name}: {e}")
    
    return metrics


def compute_connected_components(model, threshold=0.9, layer_names=None):
    """
    Compute the number of connected components in the feature space.
    
    Parameters:
        model (nn.Module): The model to analyze
        threshold (float): Threshold for feature correlation to consider connected
        layer_names (list, optional): List of layer names to analyze
        
    Returns:
        dict: Dictionary of connected components by layer
    """
    metrics = {}
    
    # Collect activations for each layer
    if hasattr(model, 'stored_activations'):
        activations = model.stored_activations
        
        for layer_name, activation in activations.items():
            # Filter by layer names if provided
            if layer_names is not None and not any(name in layer_name for name in layer_names):
                continue
            
            # Skip if not a tensor or wrong shape
            if not isinstance(activation, torch.Tensor) or activation.dim() < 2:
                continue
            
            # Reshape activations to 2D [samples, features]
            if activation.dim() > 2:
                # For convolutional activations [B, C, H, W]
                batch_size = activation.shape[0]
                act_flat = activation.transpose(0, 1).reshape(activation.shape[1], -1).T
            else:
                # For linear activations [B, F]
                act_flat = activation
            
            # Compute correlation matrix
            try:
                # Normalize each feature
                act_norm = act_flat - act_flat.mean(dim=0, keepdim=True)
                act_norm = act_norm / (act_norm.norm(dim=0, keepdim=True) + 1e-8)
                
                # Compute correlation matrix
                corr = torch.mm(act_norm.T, act_norm) / act_norm.shape[0]
                
                # Create adjacency matrix using threshold
                adj = (corr.abs() > threshold).float()
                
                # Get degrees and Laplacian
                degrees = adj.sum(dim=1)
                D = torch.diag(degrees)
                L = D - adj
                
                # Compute eigenvalues (smallest ones correspond to connected components)
                eigenvalues = torch.linalg.eigvalsh(L)
                
                # Count eigenvalues close to zero (within numerical precision)
                num_components = (eigenvalues < 1e-5).sum().item()
                
                # Store metrics
                metrics[f"{layer_name}_components"] = num_components
                metrics[f"{layer_name}_avg_corr"] = corr.abs().mean().item()
                
            except Exception as e:
                print(f"Error computing components for {layer_name}: {e}")
    
    return metrics


def get_layer_activations(model, data_loader, layer_names=None, device=None, max_batches=1):
    """
    Collect activations for specific layers of the model.
    
    Parameters:
        model (nn.Module): The model
        data_loader (DataLoader): Data loader
        layer_names (list, optional): List of layer names to collect activations for
        device (torch.device, optional): Device to use
        max_batches (int): Maximum number of batches to process
        
    Returns:
        dict: Dictionary mapping layer names to activations
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Enable activation recording if available
    if hasattr(model, 'record_activations'):
        old_record_setting = model.record_activations
        model.record_activations = True
    
    activations = {}
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            inputs = inputs.to(device)
            
            # Check if model's forward accepts store_activations parameter
            if hasattr(model, 'forward') and 'store_activations' in model.forward.__code__.co_varnames:
                outputs = model(inputs, store_activations=True)
                if isinstance(outputs, tuple):
                    outputs, batch_activations = outputs
                    
                    # Filter by layer names if provided
                    if layer_names is not None:
                        filtered_activations = {}
                        for name, activation in batch_activations.items():
                            if any(layer_name in name for layer_name in layer_names):
                                filtered_activations[name] = activation
                        batch_activations = filtered_activations
                    
                    # Store activations
                    for name, activation in batch_activations.items():
                        if name not in activations:
                            activations[name] = []
                        activations[name].append(activation)
            else:
                # Just run forward pass, model might store activations internally
                _ = model(inputs)
                
                # Check if model has stored activations
                if hasattr(model, 'stored_activations'):
                    batch_activations = model.stored_activations
                    
                    # Filter by layer names if provided
                    if layer_names is not None:
                        filtered_activations = {}
                        for name, activation in batch_activations.items():
                            if any(layer_name in name for layer_name in layer_names):
                                filtered_activations[name] = activation
                        batch_activations = filtered_activations
                    
                    # Store activations
                    for name, activation in batch_activations.items():
                        if name not in activations:
                            activations[name] = []
                        activations[name].append(activation)
    
    # Restore original activation recording setting
    if hasattr(model, 'record_activations'):
        model.record_activations = old_record_setting
    
    # Concatenate activations from different batches
    for name in activations:
        if isinstance(activations[name][0], torch.Tensor):
            activations[name] = torch.cat(activations[name], dim=0)
    
    return activations