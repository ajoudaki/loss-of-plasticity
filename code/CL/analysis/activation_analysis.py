"""
Analysis tools for examining neural network activations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import entropy, kurtosis, skew

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def analyze_activations(activations, layer_name=None):
    """
    Analyze activations for a specific layer or all layers.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_name (str, optional): Name of layer to analyze. If None, analyze all layers.
        
    Returns:
        dict: Dictionary of activation statistics by layer
    """
    results = {}
    
    if layer_name is not None and layer_name in activations:
        # Analyze single layer
        layer_activations = activations[layer_name]
        results[layer_name] = compute_activation_statistics(layer_activations)
    else:
        # Analyze all layers
        for name, layer_activations in activations.items():
            results[name] = compute_activation_statistics(layer_activations)
    
    return results


def compute_activation_statistics(activations):
    """
    Compute statistics for a single layer's activations.
    
    Parameters:
        activations (torch.Tensor): Activation tensor with shape [batch_size, ...features...]
        
    Returns:
        dict: Dictionary of activation statistics
    """
    # Ensure activations is a tensor
    if not isinstance(activations, torch.Tensor):
        return {}
    
    # Reshape activations to 2D [samples, features] if not already
    if activations.dim() > 2:
        act_flat = activations.reshape(activations.shape[0], -1)
    else:
        act_flat = activations
    
    # Move to CPU for statistics computation
    act_np = act_flat.detach().cpu().numpy()
    
    # Compute basic statistics
    stats = {
        'mean': float(np.mean(act_np)),
        'std': float(np.std(act_np)),
        'min': float(np.min(act_np)),
        'max': float(np.max(act_np)),
        'median': float(np.median(act_np)),
        'samples': act_np.shape[0],
        'features': act_np.shape[1]
    }
    
    # Compute advanced statistics
    try:
        # Fraction of dead neurons (zero activations)
        dead_threshold = 1e-9
        zero_fraction = np.mean(np.abs(act_np) < dead_threshold)
        stats['dead_fraction'] = float(zero_fraction)
        
        # Compute skewness and kurtosis
        stats['skewness'] = float(skew(act_np.flatten()))
        stats['kurtosis'] = float(kurtosis(act_np.flatten()))
        
        # Compute entropy of feature activations
        feature_means = np.mean(np.abs(act_np), axis=0)
        if np.sum(feature_means) > 0:
            feature_probs = feature_means / np.sum(feature_means)
            stats['feature_entropy'] = float(entropy(feature_probs))
        else:
            stats['feature_entropy'] = 0.0
            
        # Check for saturated neurons
        saturated_threshold = 0.99  # Consider >0.99 as saturated for normalized activations
        if np.max(act_np) <= 1.0:
            # Activations might be normalized (e.g., sigmoid, tanh outputs)
            saturated_fraction = np.mean(np.abs(act_np) > saturated_threshold)
            stats['saturated_fraction'] = float(saturated_fraction)
        
        # Feature diversity/redundancy - cosine similarity
        if act_np.shape[0] > 1 and act_np.shape[1] > 1:
            # Compute feature correlations (normalized dot product)
            norms = np.linalg.norm(act_np, axis=0, keepdims=True)
            normalized_acts = act_np / (norms + 1e-8)
            similarity_matrix = np.abs(np.corrcoef(normalized_acts.T))
            
            # Compute average off-diagonal similarity
            mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
            avg_similarity = np.mean(similarity_matrix[mask])
            stats['avg_feature_similarity'] = float(avg_similarity)
            
            # Count highly correlated features
            high_corr_threshold = 0.8
            high_corr_pairs = np.sum(similarity_matrix[mask] > high_corr_threshold)
            stats['high_correlation_pairs'] = int(high_corr_pairs)
    except Exception as e:
        print(f"Error computing advanced statistics: {e}")
    
    return stats


def visualize_activation_distributions(activations, layer_names=None, figsize=(12, 8), save_path=None):
    """
    Visualize activation distributions for selected layers.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_names (list, optional): List of layer names to visualize. If None, use all layers.
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure. If None, display the figure.
        
    Returns:
        plt.Figure: The generated figure
    """
    if layer_names is None:
        layer_names = list(activations.keys())
    
    # Create figure
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, 2, figsize=figsize)
    
    # Single layer case
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for i, layer_name in enumerate(layer_names):
        if layer_name not in activations:
            continue
        
        # Get activations for this layer
        layer_activations = activations[layer_name]
        
        # Skip if not a tensor
        if not isinstance(layer_activations, torch.Tensor):
            continue
        
        # Reshape activations to 2D [samples, features] if not already
        if layer_activations.dim() > 2:
            act_flat = layer_activations.reshape(layer_activations.shape[0], -1)
        else:
            act_flat = layer_activations
        
        # Convert to numpy
        act_np = act_flat.detach().cpu().numpy()
        
        # 1. Histogram of activation values
        axes[i, 0].hist(act_np.flatten(), bins=50, alpha=0.7)
        axes[i, 0].set_title(f'{layer_name} - Activation Distribution')
        axes[i, 0].set_xlabel('Activation Value')
        axes[i, 0].set_ylabel('Frequency')
        
        # 2. Activation patterns across neurons
        if act_np.shape[1] <= 100:  # Limit to avoid overcrowding
            # Plot mean absolute activation per neuron
            neuron_means = np.mean(np.abs(act_np), axis=0)
            axes[i, 1].bar(range(len(neuron_means)), neuron_means)
            axes[i, 1].set_title(f'{layer_name} - Mean Neuron Activity')
            axes[i, 1].set_xlabel('Neuron Index')
            axes[i, 1].set_ylabel('Mean Absolute Activation')
        else:
            # For large layers, show distribution of mean activations
            neuron_means = np.mean(np.abs(act_np), axis=0)
            axes[i, 1].hist(neuron_means, bins=50)
            axes[i, 1].set_title(f'{layer_name} - Distribution of Neuron Activity')
            axes[i, 1].set_xlabel('Mean Absolute Activation')
            axes[i, 1].set_ylabel('Number of Neurons')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def visualize_feature_space(activations, layer_name, n_components=2, method='tsne', figsize=(10, 8),
                           targets=None, save_path=None):
    """
    Visualize the feature space using dimensionality reduction.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_name (str): Name of layer to visualize
        n_components (int): Number of components for dimensionality reduction
        method (str): 'pca' or 'tsne'
        figsize (tuple): Figure size
        targets (torch.Tensor, optional): Target labels for coloring points
        save_path (str, optional): Path to save the figure. If None, display the figure.
        
    Returns:
        plt.Figure: The generated figure
    """
    if layer_name not in activations:
        print(f"Layer {layer_name} not found in activations")
        return None
    
    # Get activations for this layer
    layer_activations = activations[layer_name]
    
    # Reshape activations to 2D [samples, features] if not already
    if layer_activations.dim() > 2:
        act_flat = layer_activations.reshape(layer_activations.shape[0], -1)
    else:
        act_flat = layer_activations
    
    # Convert to numpy
    act_np = act_flat.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    else:  # Default to t-SNE
        reducer = TSNE(n_components=n_components, random_state=42)
    
    # Reduce dimensionality
    reduced_data = reducer.fit_transform(act_np)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot reduced data
    if targets is not None:
        # Convert targets to numpy if needed
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Use targets for coloring
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=targets, 
                          cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
    
    ax.set_title(f'{layer_name} - Feature Space ({method.upper()})')
    ax.set_xlabel(f'Component 1')
    ax.set_ylabel(f'Component 2')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def visualize_feature_correlations(activations, layer_name, max_features=50, figsize=(10, 8), save_path=None):
    """
    Visualize feature correlations as a heatmap.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_name (str): Name of layer to visualize
        max_features (int): Maximum number of features to include
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure. If None, display the figure.
        
    Returns:
        plt.Figure: The generated figure
    """
    if not SEABORN_AVAILABLE:
        print("Seaborn is required for correlation heatmaps")
        return None
    
    if layer_name not in activations:
        print(f"Layer {layer_name} not found in activations")
        return None
    
    # Get activations for this layer
    layer_activations = activations[layer_name]
    
    # Reshape activations to 2D [samples, features] if not already
    if layer_activations.dim() > 2:
        act_flat = layer_activations.reshape(layer_activations.shape[0], -1)
    else:
        act_flat = layer_activations
    
    # Convert to numpy
    act_np = act_flat.detach().cpu().numpy()
    
    # Limit number of features if needed
    if act_np.shape[1] > max_features:
        # Select features with highest variance
        variances = np.var(act_np, axis=0)
        top_indices = np.argsort(-variances)[:max_features]
        act_np = act_np[:, top_indices]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(act_np.T)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot correlation heatmap
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm', 
               square=True, ax=ax)
    
    ax.set_title(f'{layer_name} - Feature Correlations')
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def compare_activations(activations1, activations2, layer_names=None, metric='cosine', figsize=(12, 8), save_path=None):
    """
    Compare activations between two models or states.
    
    Parameters:
        activations1 (dict): Dictionary mapping layer names to activation tensors (model/state 1)
        activations2 (dict): Dictionary mapping layer names to activation tensors (model/state 2)
        layer_names (list, optional): List of layer names to compare. If None, use common layers.
        metric (str): Similarity metric ('cosine', 'l2', 'correlation')
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure. If None, display the figure.
        
    Returns:
        plt.Figure: The generated figure
    """
    # Find common layers if none specified
    if layer_names is None:
        layer_names = [name for name in activations1 if name in activations2]
    
    # Create figure
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    
    # Single layer case
    if n_layers == 1:
        axes = [axes]
    
    similarities = {}
    
    for i, layer_name in enumerate(layer_names):
        if layer_name not in activations1 or layer_name not in activations2:
            continue
        
        # Get activations for this layer
        acts1 = activations1[layer_name]
        acts2 = activations2[layer_name]
        
        # Skip if not tensors with matching shape
        if not isinstance(acts1, torch.Tensor) or not isinstance(acts2, torch.Tensor):
            continue
        
        if acts1.shape != acts2.shape:
            print(f"Shape mismatch for {layer_name}: {acts1.shape} vs {acts2.shape}")
            continue
        
        # Reshape activations to 2D [samples, features] if not already
        if acts1.dim() > 2:
            acts1_flat = acts1.reshape(acts1.shape[0], -1)
            acts2_flat = acts2.reshape(acts2.shape[0], -1)
        else:
            acts1_flat = acts1
            acts2_flat = acts2
        
        # Convert to numpy
        acts1_np = acts1_flat.detach().cpu().numpy()
        acts2_np = acts2_flat.detach().cpu().numpy()
        
        # Compute similarity
        if metric == 'cosine':
            # Compute cosine similarity for each sample
            norms1 = np.linalg.norm(acts1_np, axis=1, keepdims=True)
            norms2 = np.linalg.norm(acts2_np, axis=1, keepdims=True)
            normalized1 = acts1_np / (norms1 + 1e-8)
            normalized2 = acts2_np / (norms2 + 1e-8)
            similarity = np.sum(normalized1 * normalized2, axis=1)
        elif metric == 'l2':
            # Compute negative L2 distance (higher is more similar)
            similarity = -np.sqrt(np.sum((acts1_np - acts2_np)**2, axis=1))
        elif metric == 'correlation':
            # Compute correlation for each sample
            similarity = np.zeros(acts1_np.shape[0])
            for j in range(acts1_np.shape[0]):
                corr = np.corrcoef(acts1_np[j], acts2_np[j])[0, 1]
                similarity[j] = corr if not np.isnan(corr) else 0
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Store similarity
        similarities[layer_name] = similarity
        
        # Plot histogram of similarities
        axes[i].hist(similarity, bins=30, alpha=0.7)
        axes[i].set_title(f'{layer_name} - Activation Similarity ({metric})')
        axes[i].set_xlabel('Similarity')
        axes[i].set_ylabel('Frequency')
        
        # Add mean similarity
        mean_sim = np.mean(similarity)
        axes[i].axvline(mean_sim, color='r', linestyle='--', 
                      label=f'Mean: {mean_sim:.3f}')
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig, similarities