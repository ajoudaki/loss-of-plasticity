"""
Visualization utilities for continual learning experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def plot_learning_curves(trainer_history, figsize=(12, 8), save_path=None):
    """
    Plot learning curves from training history.
    
    Parameters:
        trainer_history (dict): Training history from Trainer
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss curves
    epochs = np.arange(1, len(trainer_history['train_loss']) + 1)
    ax1.plot(epochs, trainer_history['train_loss'], label='Train Loss')
    
    if 'val_loss' in trainer_history:
        ax1.plot(epochs, trainer_history['val_loss'], label='Validation Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    ax2.plot(epochs, trainer_history['train_acc'], label='Train Accuracy')
    
    if 'val_acc' in trainer_history:
        ax2.plot(epochs, trainer_history['val_acc'], label='Validation Accuracy')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_task_performance(task_accuracies, figsize=(10, 6), save_path=None):
    """
    Plot performance across tasks.
    
    Parameters:
        task_accuracies (dict): Dictionary mapping task names to accuracies
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    # Filter out non-task entries (like 'average')
    task_names = [name for name in task_accuracies.keys() if name.startswith('task_')]
    task_names.sort(key=lambda x: int(x.split('_')[1]))
    
    accuracies = [task_accuracies[name] for name in task_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot task accuracies
    ax.bar(task_names, accuracies, alpha=0.7)
    
    # Add average line if available
    if 'average' in task_accuracies:
        ax.axhline(y=task_accuracies['average'], color='r', linestyle='--', 
                  label=f"Average: {task_accuracies['average']:.2f}")
        ax.legend()
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance Across Tasks')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0 and have a reasonable upper limit
    ax.set_ylim(0, max(1.0, max(accuracies) * 1.1))
    
    # Add value labels on each bar
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_forgetting_curves(history, figsize=(12, 8), save_path=None):
    """
    Plot forgetting curves across tasks.
    
    Parameters:
        history (dict): Training history from ContinualTrainer
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    if 'tasks' not in history:
        print("No task history found")
        return None
    
    # Get task history
    tasks = history['tasks']
    
    # Initialize task accuracies dictionary
    task_accuracies = {}
    
    # Extract task accuracies over time
    for task_history in tasks:
        task_idx = task_history['task_idx']
        
        # Skip if no task accuracies
        if 'task_accuracies' not in task_history:
            continue
        
        # For each task, get accuracy on all previous tasks
        for task_key, acc in task_history['task_accuracies'].items():
            if task_key.startswith('task_'):
                prev_task_idx = int(task_key.split('_')[1])
                
                # Skip if this is a future task
                if prev_task_idx > task_idx:
                    continue
                
                # Initialize task accuracy list if not exists
                if task_key not in task_accuracies:
                    task_accuracies[task_key] = []
                
                # Add accuracy
                task_accuracies[task_key].append(acc)
    
    # Sort task keys
    task_keys = sorted(task_accuracies.keys(), key=lambda x: int(x.split('_')[1]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot forgetting curves
    x = list(range(len(tasks)))
    x_labels = [f"After Task {i}" for i in range(len(tasks))]
    
    for task_key in task_keys:
        task_idx = int(task_key.split('_')[1])
        
        # Get accuracies, padding with None for tasks that haven't been trained yet
        accs = task_accuracies[task_key]
        padded_accs = [None] * task_idx + accs
        
        # Plot line
        ax.plot(x[task_idx:], padded_accs[task_idx:], marker='o', label=f'Task {task_idx}')
    
    ax.set_xlabel('Training Progress')
    ax.set_ylabel('Accuracy')
    ax.set_title('Task Performance Over Time (Forgetting Curves)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_activation_heatmap(activations, layer_name, n_samples=10, n_features=50, figsize=(12, 8), save_path=None):
    """
    Plot activations as a heatmap.
    
    Parameters:
        activations (dict): Dictionary mapping layer names to activation tensors
        layer_name (str): Name of the layer to visualize
        n_samples (int): Number of samples to include
        n_features (int): Number of features to include
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    if not SEABORN_AVAILABLE:
        print("Seaborn is required for heatmap visualization")
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
    
    # Move to CPU and convert to numpy
    act_np = act_flat.detach().cpu().numpy()
    
    # Limit number of samples and features
    n_samples = min(n_samples, act_np.shape[0])
    n_features = min(n_features, act_np.shape[1])
    
    # Select samples and features
    indices = np.random.choice(act_np.shape[0], n_samples, replace=False)
    
    # Select features with highest variance
    variances = np.var(act_np, axis=0)
    top_features = np.argsort(-variances)[:n_features]
    
    # Create heatmap data
    heatmap_data = act_np[indices][:, top_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(heatmap_data, cmap='viridis', ax=ax)
    
    ax.set_title(f'{layer_name} - Activation Heatmap')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Sample Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_rank_trend(model_checkpoints, layer_name, metric='effective_rank', figsize=(10, 6), save_path=None):
    """
    Plot trend of rank metrics for a layer across checkpoints.
    
    Parameters:
        model_checkpoints (list): List of (epoch, model) tuples
        layer_name (str): Name of the layer to analyze
        metric (str): Rank metric to plot ('effective_rank', 'stable_rank', 'rank_95')
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    from analysis.rank_analysis import compute_layer_ranks
    
    # Initialize lists to store data
    epochs = []
    metric_values = []
    
    # Process each checkpoint
    for epoch, model in model_checkpoints:
        # Compute ranks
        ranks = compute_layer_ranks(model, [layer_name])
        
        # Skip if layer not found
        if layer_name not in ranks:
            continue
        
        # Get metric value
        if metric in ranks[layer_name]:
            epochs.append(epoch)
            metric_values.append(ranks[layer_name][metric])
    
    # Skip if no data collected
    if not epochs:
        print(f"No rank data collected for layer {layer_name}")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot trend
    ax.plot(epochs, metric_values, marker='o')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Trend for {layer_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def visualize_weight_matrices(model, layer_names=None, figsize=(12, 8), cmap='viridis', save_path=None):
    """
    Visualize weight matrices for specified layers.
    
    Parameters:
        model (nn.Module): The model to visualize
        layer_names (list, optional): List of layer names to visualize
        figsize (tuple): Figure size
        cmap (str): Colormap to use
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    # If no layers specified, find all linear layers
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                layer_names.append(name)
    
    # Limit to first 4 layers to avoid too many plots
    if len(layer_names) > 4:
        layer_names = layer_names[:4]
    
    # Create figure
    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    
    # Handle single layer case
    if n_layers == 1:
        axes = [axes]
    
    # Visualize each layer's weights
    for ax, name in zip(axes, layer_names):
        for module_name, module in model.named_modules():
            if module_name == name and hasattr(module, 'weight'):
                # Get weight matrix
                weight = module.weight.data.detach().cpu()
                
                # For convolutional layers, reshape
                if isinstance(module, torch.nn.Conv2d):
                    weight = weight.view(weight.size(0), -1)
                
                # If weight is too large, sample or resize
                if weight.shape[0] > 100 or weight.shape[1] > 100:
                    if weight.shape[0] > weight.shape[1]:
                        # More rows than columns, sample rows
                        indices = torch.linspace(0, weight.shape[0]-1, 100).long()
                        weight = weight[indices]
                    else:
                        # More columns than rows, sample columns
                        indices = torch.linspace(0, weight.shape[1]-1, 100).long()
                        weight = weight[:, indices]
                
                # Plot weight matrix
                im = ax.imshow(weight, aspect='auto', cmap=cmap)
                ax.set_title(f'{name} ({weight.shape[0]}×{weight.shape[1]})')
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_activation_evolution(activation_history, layer_name, feature_index=None, figsize=(12, 8), save_path=None):
    """
    Plot how activations for a specific layer evolve over training.
    
    Parameters:
        activation_history (dict): Dictionary mapping epochs to activation dictionaries
        layer_name (str): Name of the layer to visualize
        feature_index (int, optional): Index of feature to track. If None, use mean across features.
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: The generated figure
    """
    # Extract epochs and ensure they're sorted
    epochs = sorted([int(k.split('_')[1]) for k in activation_history.keys() if k.startswith('epoch_')])
    
    # Check if layer exists in all epochs
    for epoch in epochs:
        epoch_key = f'epoch_{epoch}'
        if epoch_key not in activation_history or layer_name not in activation_history[epoch_key]:
            print(f"Layer {layer_name} not found in epoch {epoch}")
            return None
    
    # Extract activation statistics for each epoch
    means = []
    stds = []
    max_vals = []
    min_vals = []
    
    for epoch in epochs:
        epoch_key = f'epoch_{epoch}'
        act = activation_history[epoch_key][layer_name]
        
        # Reshape if needed
        if act.dim() > 2:
            act = act.reshape(act.shape[0], -1)
        
        # Extract statistics for specific feature or mean across features
        if feature_index is not None:
            # Check if feature index is valid
            if feature_index >= act.shape[1]:
                print(f"Feature index {feature_index} out of range for layer {layer_name}")
                return None
            
            feature_acts = act[:, feature_index]
            means.append(feature_acts.mean().item())
            stds.append(feature_acts.std().item())
            max_vals.append(feature_acts.max().item())
            min_vals.append(feature_acts.min().item())
        else:
            # Mean across all features
            means.append(act.mean().item())
            stds.append(act.std().item())
            max_vals.append(act.max().item())
            min_vals.append(act.min().item())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot mean with std deviation as shaded region
    ax.plot(epochs, means, 'b-', label='Mean')
    ax.fill_between(epochs, [m - s for m, s in zip(means, stds)], 
                  [m + s for m, s in zip(means, stds)], alpha=0.3, color='b')
    
    # Plot min and max
    ax.plot(epochs, min_vals, 'r--', label='Min')
    ax.plot(epochs, max_vals, 'g--', label='Max')
    
    # Add title and labels
    title = f'{layer_name} - Activation Evolution'
    if feature_index is not None:
        title += f' (Feature {feature_index})'
    else:
        title += ' (All Features)'
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Activation Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


def plot_singular_values_spectrum(rank_data, layer_name, epoch_indices=None, figsize=(10, 6), log_scale=True, save_path=None):
    """
    Plot the singular value spectrum for a given layer over selected epochs.

    Parameters:
        rank_data (dict): Dictionary with keys 'epochs' (a list of epochs) and 'layers'.
                          Each entry in 'layers' is a dict mapping a layer name to rank metrics,
                          including 'singular_values' (a list of singular value arrays over epochs).
        layer_name (str): Name of the layer to analyze.
        epoch_indices (list, optional): List of epoch indices to plot. If None, defaults to first, middle, and last epochs.
        figsize (tuple): Size of the figure.
        log_scale (bool): Whether to use a logarithmic scale for the y–axis.
        save_path (str, optional): If provided, the plot is saved to this file.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    if layer_name not in rank_data['layers']:
        print(f"Layer '{layer_name}' not found in rank data.")
        return None

    epochs = rank_data['epochs']
    singular_values = rank_data['layers'][layer_name]['singular_values']

    # Determine which epochs to plot if not provided.
    if epoch_indices is None:
        n_epochs = len(epochs)
        if n_epochs <= 3:
            epoch_indices = list(range(n_epochs))
        else:
            epoch_indices = [0, n_epochs // 2, n_epochs - 1]

    fig, ax = plt.subplots(figsize=figsize)
    for idx in epoch_indices:
        if idx < len(epochs) and idx < len(singular_values):
            sv = singular_values[idx]
            epoch = epochs[idx]
            ax.plot(np.arange(1, len(sv) + 1), sv, marker='o', label=f'Epoch {epoch}')
    
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"Singular Value Spectrum for '{layer_name}'")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    
    return fig
