import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_metrics_history(history, metric_name, title=None, figsize=(14, 8)):
    """
    Plot the history of metrics over tasks and epochs.
    
    Args:
        history: Dictionary containing task training history
        metric_name: Name of the metric to plot
        title: Optional title for the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    for task_id, task_info in history['tasks'].items():
        task_history = task_info['history']
        epochs = task_history['epochs']
        
        # Plot current task metrics
        plt.plot(epochs, task_history['current'][metric_name], 
                marker='o', linestyle='-', 
                label=f'Task {task_id} (Classes {task_info["classes"]})')
        
        # Plot old task metrics if available
        if 'old' in task_history and len(task_history['old'][metric_name]) > 0:
            plt.plot(epochs, task_history['old'][metric_name],
                    marker='x', linestyle='--',
                    label=f'Old Classes after Task {task_id}')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_', ' ').title())
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric_name.replace("_", " ").title()} During Continual Learning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_layer_metrics(metrics_dict, metric_name, title=None, figsize=(14, 8)):
    """
    Plot metrics for different layers.
    
    Args:
        metrics_dict: Dictionary mapping layer_name -> metrics
        metric_name: Name of the metric to plot
        title: Optional title for the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    # Extract layer names and metric values
    layer_names = list(metrics_dict.keys())
    metric_values = [metrics_dict[layer][metric_name] for layer in layer_names]
    
    # Create bar plot
    y_pos = np.arange(len(layer_names))
    plt.barh(y_pos, metric_values)
    plt.yticks(y_pos, layer_names)
    plt.xlabel(metric_name.replace('_', ' ').title())
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric_name.replace("_", " ").title()} by Layer')
    
    plt.tight_layout()
    return plt.gcf()

def plot_layer_metrics_over_time(history, layer_name, metric_name, title=None, figsize=(14, 8)):
    """
    Plot how a specific layer metric changes over time during training.
    
    Args:
        history: Dictionary containing task training history
        layer_name: Name of the layer to plot metrics for
        metric_name: Name of the metric to plot
        title: Optional title for the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    for task_id, task_info in history['tasks'].items():
        task_history = task_info['history']
        
        if layer_name in task_history['training_metrics_history']:
            # Only include epochs where we collected metrics
            epochs = [ep for i, ep in enumerate(task_history['epochs']) 
                     if i < len(task_history['training_metrics_history'][layer_name][metric_name])]
            
            train_values = task_history['training_metrics_history'][layer_name][metric_name]
            val_values = task_history['validation_metrics_history'][layer_name][metric_name]
            
            plt.plot(epochs, train_values, 
                    marker='o', linestyle='-', 
                    label=f'Task {task_id} Train')
            
            plt.plot(epochs, val_values, 
                    marker='x', linestyle='--', 
                    label=f'Task {task_id} Val')
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.replace('_', ' ').title())
    
    if title:
        plt.title(title)
    else:
        plt.title(f'{metric_name.replace("_", " ").title()} for Layer {layer_name}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def visualize_activations(activations, n_cols=5, figsize=(15, 10)):
    """
    Visualize activation patterns for a layer.
    
    Args:
        activations: Tensor of activations [batch_size, features]
        n_cols: Number of columns for the visualization grid
        figsize: Figure size tuple
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    
    batch_size, n_features = activations.shape
    sample_idx = np.random.choice(batch_size, min(16, batch_size), replace=False)
    
    activations = activations[sample_idx]
    
    n_rows = (len(sample_idx) + n_cols - 1) // n_cols
    plt.figure(figsize=figsize)
    
    for i, idx in enumerate(sample_idx):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(activations[i].reshape(-1, 1), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Sample {i}')
        plt.tight_layout()
    
    return plt.gcf()