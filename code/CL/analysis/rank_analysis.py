"""
Analysis tools for examining rank and dimensionality in neural networks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svdvals
from sklearn.decomposition import PCA


def compute_layer_ranks(model, layer_names=None):
    """
    Compute the rank of weight matrices for specified layers.
    
    Parameters:
        model (nn.Module): The model to analyze
        layer_names (list, optional): List of layer names to analyze. If None, analyze all linear layers.
        
    Returns:
        dict: Dictionary mapping layer names to rank metrics
    """
    results = {}
    
    # If no layers specified, find all linear layers
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                layer_names.append(name)
    
    # Compute rank for each layer
    for name, module in model.named_modules():
        if name in layer_names:
            if hasattr(module, 'weight'):
                weight = module.weight.data
                
                # Skip if weight is None
                if weight is None:
                    continue
                
                # Reshape weight matrix for rank computation
                if isinstance(module, torch.nn.Conv2d):
                    # Reshape conv weights for rank computation
                    w_reshape = weight.view(weight.size(0), -1)
                else:
                    w_reshape = weight
                
                # Move to CPU for computation
                w_np = w_reshape.detach().cpu().numpy()
                
                # Compute full rank
                full_rank = min(w_np.shape)
                
                # Compute effective rank based on SVD
                try:
                    s = svdvals(w_np)
                    s_norm = s / np.sum(s)
                    entropy = -np.sum(s_norm * np.log(s_norm + 1e-10))
                    eff_rank = np.exp(entropy)
                    
                    # Compute rank at different thresholds
                    rank_99 = np.sum(np.cumsum(s) / np.sum(s) < 0.99) + 1
                    rank_95 = np.sum(np.cumsum(s) / np.sum(s) < 0.95) + 1
                    rank_90 = np.sum(np.cumsum(s) / np.sum(s) < 0.90) + 1
                    
                    # Compute stable rank (nuclear norm squared / operator norm squared)
                    stable_rank = np.sum(s)**2 / (s[0]**2)
                    
                    results[name] = {
                        'full_rank': full_rank,
                        'effective_rank': eff_rank,
                        'rank_99': rank_99,
                        'rank_95': rank_95,
                        'rank_90': rank_90,
                        'stable_rank': stable_rank,
                        'singular_values': s,
                        'shape': w_np.shape
                    }
                except Exception as e:
                    print(f"Error computing rank for {name}: {e}")
    
    return results


def compute_feature_diversity(activations, layer_names=None):
    """
    Compute metrics related to feature diversity and redundancy.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_names (list, optional): List of layer names to analyze. If None, analyze all layers.
        
    Returns:
        dict: Dictionary mapping layer names to diversity metrics
    """
    results = {}
    
    # If no layers specified, use all available layers
    if layer_names is None:
        layer_names = list(activations.keys())
    
    # Compute diversity for each layer
    for name in layer_names:
        if name not in activations:
            continue
        
        layer_activations = activations[name]
        
        # Skip if not a tensor
        if not isinstance(layer_activations, torch.Tensor):
            continue
        
        # Reshape activations to 2D [samples, features] if not already
        if layer_activations.dim() > 2:
            act_flat = layer_activations.reshape(layer_activations.shape[0], -1)
        else:
            act_flat = layer_activations
        
        # Move to CPU for computation
        act_np = act_flat.detach().cpu().numpy()
        
        # Compute mean activation per feature
        feature_means = np.mean(np.abs(act_np), axis=0)
        
        # Compute entropy of feature activations (as a diversity measure)
        if np.sum(feature_means) > 0:
            feature_probs = feature_means / np.sum(feature_means)
            feature_entropy = -np.sum(feature_probs * np.log(feature_probs + 1e-10))
            normalized_entropy = feature_entropy / np.log(len(feature_means) + 1e-10)
        else:
            feature_entropy = 0.0
            normalized_entropy = 0.0
        
        # Compute feature correlations
        try:
            # Normalize features
            act_norm = act_np - np.mean(act_np, axis=0, keepdims=True)
            norms = np.linalg.norm(act_norm, axis=0, keepdims=True)
            act_norm = act_norm / (norms + 1e-8)
            
            # Compute correlation matrix
            corr_matrix = np.abs(np.corrcoef(act_norm.T))
            np.fill_diagonal(corr_matrix, 0)  # Zero out diagonal
            
            # Compute average correlation
            avg_correlation = np.mean(corr_matrix)
            
            # Count highly correlated pairs
            high_corr_pairs = np.sum(corr_matrix > 0.8) / 2  # Divide by 2 for unique pairs
            high_corr_fraction = high_corr_pairs / (len(corr_matrix) * (len(corr_matrix) - 1) / 2)
            
            # Compute effective dimensionality using PCA
            if act_np.shape[0] > 1 and act_np.shape[1] > 1:
                pca = PCA().fit(act_np)
                explained_variance = pca.explained_variance_ratio_
                
                # Compute number of components needed for different variance thresholds
                dim_99 = np.sum(np.cumsum(explained_variance) < 0.99) + 1
                dim_95 = np.sum(np.cumsum(explained_variance) < 0.95) + 1
                dim_90 = np.sum(np.cumsum(explained_variance) < 0.90) + 1
                
                # Compute participation ratio
                participation_ratio = np.sum(explained_variance)**2 / np.sum(explained_variance**2)
            else:
                dim_99 = dim_95 = dim_90 = participation_ratio = 0
        except Exception as e:
            print(f"Error computing correlations for {name}: {e}")
            avg_correlation = high_corr_pairs = high_corr_fraction = 0
            dim_99 = dim_95 = dim_90 = participation_ratio = 0
        
        # Store results
        results[name] = {
            'feature_entropy': feature_entropy,
            'normalized_entropy': normalized_entropy,
            'avg_correlation': avg_correlation,
            'high_corr_pairs': high_corr_pairs,
            'high_corr_fraction': high_corr_fraction,
            'pca_dim_99': dim_99,
            'pca_dim_95': dim_95,
            'pca_dim_90': dim_90,
            'participation_ratio': participation_ratio
        }
    
    return results


def analyze_rank_dynamics(model_checkpoints, layer_names=None):
    """
    Analyze how rank changes across training checkpoints.
    
    Parameters:
        model_checkpoints (list): List of (epoch, model) tuples
        layer_names (list, optional): List of layer names to analyze
        
    Returns:
        dict: Dictionary mapping layer names to rank trajectories
    """
    # Initialize results dictionary
    results = {
        'epochs': [],
        'layers': {}
    }
    
    # Process each checkpoint
    for epoch, model in model_checkpoints:
        results['epochs'].append(epoch)
        
        # Compute ranks for this checkpoint
        ranks = compute_layer_ranks(model, layer_names)
        
        # Store rank metrics for each layer
        for layer_name, layer_ranks in ranks.items():
            if layer_name not in results['layers']:
                # Initialize layer data
                results['layers'][layer_name] = {
                    'effective_rank': [],
                    'stable_rank': [],
                    'rank_95': [],
                    'singular_values': []
                }
            
            # Store rank metrics for this epoch
            results['layers'][layer_name]['effective_rank'].append(layer_ranks['effective_rank'])
            results['layers'][layer_name]['stable_rank'].append(layer_ranks['stable_rank'])
            results['layers'][layer_name]['rank_95'].append(layer_ranks['rank_95'])
            results['layers'][layer_name]['singular_values'].append(layer_ranks['singular_values'])
    
    return results


def plot_rank_dynamics(rank_dynamics, metric='effective_rank', figsize=(12, 8), save_path=None):
    """
    Plot rank dynamics over training.
    
    Parameters:
        rank_dynamics (dict): Dictionary returned by analyze_rank_dynamics
        metric (str): Rank metric to plot ('effective_rank', 'stable_rank', 'rank_95')
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    epochs = rank_dynamics['epochs']
    layers = rank_dynamics['layers']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for layer_name, layer_data in layers.items():
        if metric in layer_data:
            ax.plot(epochs, layer_data[metric], marker='o', label=layer_name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title(f'{metric.replace("_", " ").title()} Dynamics During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_singular_value_spectrum(rank_data, layer_name, epoch_indices=None, figsize=(12, 8), log_scale=True, save_path=None):
    """
    Plot singular value spectrum for a layer at different epochs.
    
    Parameters:
        rank_data (dict): Dictionary returned by analyze_rank_dynamics
        layer_name (str): Name of the layer to analyze
        epoch_indices (list, optional): Indices of epochs to include. If None, use first, middle, and last.
        figsize (tuple): Figure size
        log_scale (bool): Whether to use log scale for y-axis
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    if layer_name not in rank_data['layers']:
        print(f"Layer {layer_name} not found in rank data")
        return None
    
    epochs = rank_data['epochs']
    singular_values = rank_data['layers'][layer_name]['singular_values']
    
    # Select epochs to plot
    if epoch_indices is None:
        n_epochs = len(epochs)
        if n_epochs <= 3:
            epoch_indices = list(range(n_epochs))
        else:
            epoch_indices = [0, n_epochs // 2, n_epochs - 1]  # First, middle, last
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx in epoch_indices:
        if idx < len(epochs) and idx < len(singular_values):
            sv = singular_values[idx]
            epoch = epochs[idx]
            
            # Plot singular values
            ax.plot(np.arange(1, len(sv) + 1), sv, marker='.', label=f'Epoch {epoch}')
    
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value')
    ax.set_title(f'Singular Value Spectrum - {layer_name}')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def compute_spectral_decay_rate(singular_values):
    """
    Compute the spectral decay rate of singular values.
    
    Parameters:
        singular_values (numpy.ndarray): Array of singular values
        
    Returns:
        float: Estimated decay rate
    """
    if len(singular_values) < 2:
        return 0.0
    
    # Take log of singular values
    log_sv = np.log(singular_values + 1e-10)
    
    # Fit linear regression to log(singular_values) vs. log(index)
    indices = np.log(np.arange(1, len(singular_values) + 1))
    
    # Use polyfit to get slope
    slope, _ = np.polyfit(indices, log_sv, 1)
    
    return slope