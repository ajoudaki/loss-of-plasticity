"""
Cloning experiment for the Loss of Plasticity project.

This script implements a self-contained experiment to validate the proposition on cloning
in neural networks as described in the paper. It demonstrates how cloned networks behave
like smaller networks, confining the parameter trajectory to lower-dimensional subspaces
and potentially contributing to loss of plasticity.

The experiment trains both original and cloned networks on MNIST and CIFAR10, tracking
various metrics including test accuracy, weight norms, effective rank, and feature correlations.
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.cloning import model_clone, test_activation_cloning
from src.utils.metrics import create_module_filter
from src.utils.monitor import NetworkMonitor

# Set up the output directory for figures
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set the random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MLP model with optional normalization
class MLP(nn.Module):
    def __init__(self, 
                 input_size=784, 
                 hidden_sizes=[512, 256, 128], 
                 output_size=10, 
                 activation='relu',
                 dropout_p=0.0,
                 normalization=None):
        """Fully connected MLP with customizable architecture."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add normalization if specified
            if normalization == 'batch':
                layers.append(nn.BatchNorm1d(hidden_size))
            elif normalization == 'layer':
                layers.append(nn.LayerNorm(hidden_size))
                
            # Add activation
            layers.append(self.activation)
            
            # Add dropout if specified
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
                
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        return self.layers(x)

def create_dataset_loaders(dataset_name, batch_size=128, val_split=0.1):
    """Create data loaders for the specified dataset."""
    if dataset_name.lower() == 'mnist':
        # MNIST normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load datasets
        train_dataset = datasets.MNIST(
            root=os.path.join(PROJECT_ROOT, 'data'),
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            root=os.path.join(PROJECT_ROOT, 'data'),
            train=False,
            download=True,
            transform=transform
        )
        
        input_size = 28 * 28
        num_classes = 10
        in_channels = 1
        
    elif dataset_name.lower() == 'cifar10':
        # CIFAR-10 normalization
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = datasets.CIFAR10(
            root=os.path.join(PROJECT_ROOT, 'data'),
            train=True,
            download=True,
            transform=transform_train
        )
        
        test_dataset = datasets.CIFAR10(
            root=os.path.join(PROJECT_ROOT, 'data'),
            train=False,
            download=True,
            transform=transform_test
        )
        
        input_size = 32 * 32 * 3
        num_classes = 10
        in_channels = 3
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split training data into train/val
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    train_idx, val_idx = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2
    )
    
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Create fixed batch loaders for metrics
    fixed_train_idx = train_idx[:min(500, len(train_idx))]
    fixed_val_idx = val_idx[:min(500, len(val_idx))]
    
    fixed_train_sampler = SubsetRandomSampler(fixed_train_idx)
    fixed_val_sampler = SubsetRandomSampler(fixed_val_idx)
    
    fixed_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=fixed_train_sampler,
        shuffle=False
    )
    
    fixed_val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=fixed_val_sampler,
        shuffle=False
    )
    
    data_info = {
        'input_size': input_size,
        'num_classes': num_classes,
        'in_channels': in_channels
    }
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'fixed_train': fixed_train_loader,
        'fixed_val': fixed_val_loader
    }, data_info

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def calculate_effective_rank(activations):
    """Calculate the effective rank of a set of activations."""
    # Compute covariance matrix
    activations = activations.cpu().numpy()
    cov = np.cov(activations.T)
    
    # Compute singular values
    s = np.linalg.svd(cov, compute_uv=False)
    
    # Ensure positive values
    s = np.maximum(s, 1e-10)
    
    # Normalize singular values
    p = s / np.sum(s)
    
    # Calculate entropy
    entropy = -np.sum(p * np.log(p))
    
    # Calculate effective rank
    effective_rank = np.exp(entropy)
    
    return effective_rank

def calculate_feature_correlations(activations, topk=10):
    """Calculate feature correlation statistics"""
    # Get activations as numpy array
    act_np = activations.cpu().numpy()
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(act_np.T)
    
    # Replace NaNs with 0
    corr_matrix = np.nan_to_num(corr_matrix)
    
    # Get the uppermost triangle indices (excluding the diagonal)
    triu_indices = np.triu_indices_from(corr_matrix, k=1)
    pair_corrs = corr_matrix[triu_indices]
    
    # Calculate statistics
    avg_corr = np.mean(np.abs(pair_corrs))
    max_corr = np.max(np.abs(pair_corrs))
    
    # Get topk correlation indices
    topk_indices = np.argsort(-np.abs(pair_corrs))[:topk]
    topk_corrs = pair_corrs[topk_indices]
    
    stats = {
        'avg_correlation': avg_corr,
        'max_correlation': max_corr,
        'topk_correlations': np.abs(topk_corrs).tolist()
    }
    
    return stats

def collect_metrics(model, fixed_loader, monitor, layer_names=None):
    """Collect metrics for network analysis."""
    model.eval()
    
    # Prepare to store activations
    activations = {}
    
    # Register hooks to collect activations
    def get_activation(layer_name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            
            # For activations with spatial dimensions, flatten to 2D
            if output.dim() > 2:
                # Preserve batch dimension, flatten the rest
                activations[layer_name] = output.view(output.size(0), -1)
            else:
                activations[layer_name] = output
        return hook
    
    # Initialize hooks dict to store handles for removal
    hooks = {}
    
    # Set up hooks based on the model type
    if layer_names is None:
        # For Sequential models
        if isinstance(model, nn.Sequential) or hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
            seq = model if isinstance(model, nn.Sequential) else model.layers
            # Register hooks for linear layers followed by activation
            for i, layer in enumerate(seq):
                if isinstance(layer, nn.Linear) and i < len(seq) - 1:
                    # Check if the next layer is an activation function
                    next_layer = seq[i + 1]
                    if isinstance(next_layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
                        layer_name = f"linear_{i}"
                        hooks[layer_name] = next_layer.register_forward_hook(
                            get_activation(layer_name)
                        )
    else:
        # If specific layer names provided, use those
        for name, module in model.named_modules():
            if name in layer_names:
                hooks[name] = module.register_forward_hook(
                    get_activation(name)
                )
    
    # Process a batch through the model
    for inputs, targets in fixed_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _ = model(inputs)
        break
    
    # Calculate metrics for each collected activation
    metrics = {}
    
    for layer_name, layer_activations in activations.items():
        # Skip output layer
        if layer_name == "output":
            continue
        
        # Calculate percentage of dead neurons (for ReLU activations)
        if isinstance(model, nn.Sequential) or (hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential)):
            seq = model if isinstance(model, nn.Sequential) else model.layers
            if any(isinstance(layer, nn.ReLU) for layer in seq):
                zero_activations = (layer_activations.abs().sum(dim=0) == 0).float().mean().item()
                metrics[f"{layer_name}/dead_neurons"] = zero_activations * 100.0
        
        # Calculate effective rank
        effective_rank = calculate_effective_rank(layer_activations)
        metrics[f"{layer_name}/effective_rank"] = effective_rank
        
        # Calculate feature correlations
        corr_stats = calculate_feature_correlations(layer_activations)
        metrics[f"{layer_name}/avg_correlation"] = corr_stats["avg_correlation"]
        metrics[f"{layer_name}/max_correlation"] = corr_stats["max_correlation"]
    
    # Remove hooks
    for hook in hooks.values():
        hook.remove()
    
    # Calculate weight statistics
    weight_stats = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Calculate L2 norm
            l2_norm = torch.norm(param, p=2).item()
            weight_stats[f"{name}/l2_norm"] = l2_norm
            
            # Calculate average magnitude
            avg_mag = param.abs().mean().item()
            weight_stats[f"{name}/avg_magnitude"] = avg_mag
            
            # Count parameters
            total_params += param.numel()
    
    metrics.update(weight_stats)
    metrics["total_params"] = total_params
    
    # Return both metrics and activations
    return metrics, activations

def run_experiment(config, dataset_name='mnist'):
    """Run the cloning experiment with specified configuration."""
    print(f"\n{'='*80}")
    print(f"Running cloning experiment on {dataset_name.upper()} with:")
    print(f"Base hidden sizes: {config['base_hidden_sizes']}")
    print(f"Expansion factor: {config['expansion_factor']}x")
    print(f"Epochs: {config['epochs']}")
    print(f"{'='*80}\n")
    
    # Create data loaders
    dataloaders, data_info = create_dataset_loaders(
        dataset_name, 
        batch_size=config['batch_size']
    )
    
    # Create base model
    base_model = MLP(
        input_size=data_info['input_size'],
        hidden_sizes=config['base_hidden_sizes'],
        output_size=data_info['num_classes'],
        activation=config['activation'],
        dropout_p=config['dropout_p'],
        normalization=config['normalization']
    ).to(device)
    
    # Create expanded model
    expanded_hidden_sizes = [size * config['expansion_factor'] for size in config['base_hidden_sizes']]
    expanded_model = MLP(
        input_size=data_info['input_size'],
        hidden_sizes=expanded_hidden_sizes,
        output_size=data_info['num_classes'],
        activation=config['activation'],
        dropout_p=config['dropout_p'],
        normalization=config['normalization']
    ).to(device)
    
    # Setup for tracking metrics
    histories = {
        'base': {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': [],
            'metrics': []
        },
        'expanded': {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': [],
            'metrics': []
        },
        'cloned': {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': [],
            'metrics': [],
            'cloning_similarity': []
        }
    }
    
    # Create monitors
    module_filter = create_module_filter(['default'], 'mlp')
    base_monitor = NetworkMonitor(base_model, module_filter)
    
    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()
    
    base_optimizer = optim.Adam(
        base_model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    expanded_optimizer = optim.Adam(
        expanded_model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Train base model
    print(f"Training base model for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            base_model, 
            dataloaders['train'], 
            criterion, 
            base_optimizer, 
            device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            base_model, 
            dataloaders['val'], 
            criterion, 
            device
        )
        
        test_loss, test_acc = evaluate(
            base_model, 
            dataloaders['test'], 
            criterion, 
            device
        )
        
        # Record metrics
        histories['base']['train_loss'].append(train_loss)
        histories['base']['train_acc'].append(train_acc)
        histories['base']['val_loss'].append(val_loss)
        histories['base']['val_acc'].append(val_acc)
        histories['base']['test_loss'].append(test_loss)
        histories['base']['test_acc'].append(test_acc)
        
        # Collect additional metrics at specified intervals
        if epoch % config['metrics_interval'] == 0 or epoch == config['epochs']:
            metrics, _ = collect_metrics(
                base_model, 
                dataloaders['fixed_train'], 
                base_monitor
            )
            histories['base']['metrics'].append((epoch, metrics))
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == config['epochs']:
            print(f"Base - Epoch {epoch}/{config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
    
    # Clone the base model to create a cloned model
    print("\nCloning base model...")
    cloned_model = model_clone(base_model, expanded_model)
    
    # Test if cloning was successful
    fixed_batch, fixed_targets = next(iter(dataloaders['fixed_train']))
    fixed_batch, fixed_targets = fixed_batch.to(device), fixed_targets.to(device)
    
    train_success, train_unexplained_var = test_activation_cloning(
        base_model, 
        cloned_model, 
        fixed_batch, 
        fixed_targets, 
        model_name='mlp'
    )
    
    print(f"Cloning validation: {'successful' if train_success else 'failed'}")
    print(f"Unexplained activation variance: {np.mean(list(train_unexplained_var.values())):.4f}")
    
    # Create cloned model optimizer
    cloned_optimizer = optim.Adam(
        cloned_model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create expanded model monitor
    expanded_monitor = NetworkMonitor(expanded_model, module_filter)
    
    # Train expanded model from scratch
    print(f"\nTraining fresh expanded model for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            expanded_model, 
            dataloaders['train'], 
            criterion, 
            expanded_optimizer, 
            device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            expanded_model, 
            dataloaders['val'], 
            criterion, 
            device
        )
        
        test_loss, test_acc = evaluate(
            expanded_model, 
            dataloaders['test'], 
            criterion, 
            device
        )
        
        # Record metrics
        histories['expanded']['train_loss'].append(train_loss)
        histories['expanded']['train_acc'].append(train_acc)
        histories['expanded']['val_loss'].append(val_loss)
        histories['expanded']['val_acc'].append(val_acc)
        histories['expanded']['test_loss'].append(test_loss)
        histories['expanded']['test_acc'].append(test_acc)
        
        # Collect additional metrics at specified intervals
        if epoch % config['metrics_interval'] == 0 or epoch == config['epochs']:
            metrics, _ = collect_metrics(
                expanded_model, 
                dataloaders['fixed_train'], 
                expanded_monitor
            )
            histories['expanded']['metrics'].append((epoch, metrics))
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == config['epochs']:
            print(f"Expanded - Epoch {epoch}/{config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
    
    # Create cloned model monitor
    cloned_monitor = NetworkMonitor(cloned_model, module_filter)
    
    # Train cloned model
    print(f"\nTraining cloned model for {config['epochs']} epochs...")
    
    for epoch in range(1, config['epochs'] + 1):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            cloned_model, 
            dataloaders['train'], 
            criterion, 
            cloned_optimizer, 
            device
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            cloned_model, 
            dataloaders['val'], 
            criterion, 
            device
        )
        
        test_loss, test_acc = evaluate(
            cloned_model, 
            dataloaders['test'], 
            criterion, 
            device
        )
        
        # Record metrics
        histories['cloned']['train_loss'].append(train_loss)
        histories['cloned']['train_acc'].append(train_acc)
        histories['cloned']['val_loss'].append(val_loss)
        histories['cloned']['val_acc'].append(val_acc)
        histories['cloned']['test_loss'].append(test_loss)
        histories['cloned']['test_acc'].append(test_acc)
        
        # Collect additional metrics at specified intervals
        if epoch % config['metrics_interval'] == 0 or epoch == config['epochs']:
            metrics, _ = collect_metrics(
                cloned_model, 
                dataloaders['fixed_train'], 
                cloned_monitor
            )
            histories['cloned']['metrics'].append((epoch, metrics))
            
            # Test cloning similarity
            train_success, train_unexplained_var = test_activation_cloning(
                base_model, 
                cloned_model, 
                fixed_batch, 
                fixed_targets, 
                model_name='mlp'
            )
            
            # Store for later analysis
            histories['cloned']['cloning_similarity'].append({
                'epoch': epoch,
                'success': train_success,
                'unexplained_var': np.mean(list(train_unexplained_var.values()))
            })
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1 or epoch == config['epochs']:
            print(f"Cloned - Epoch {epoch}/{config['epochs']}, "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")
    
    print("\nExperiment completed!")
    
    # Return all histories and models
    return {
        'histories': histories,
        'models': {
            'base': base_model,
            'expanded': expanded_model,
            'cloned': cloned_model
        },
        'data_info': data_info,
        'config': config
    }

def plot_experiment_results(results, dataset_name):
    """Generate paper-ready figures from experiment results."""
    histories = results['histories']
    config = results['config']
    
    # Define colors
    colors = {
        'base': '#1f77b4',       # Blue
        'expanded': '#ff7f0e',   # Orange
        'cloned': '#2ca02c'      # Green
    }
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot test accuracy comparison
    plt.figure(figsize=(10, 6))
    for model_type, history in histories.items():
        plt.plot(
            range(1, config['epochs'] + 1),
            history['test_acc'],
            label=f"{model_type.capitalize()} Model",
            color=colors[model_type],
            linewidth=2
        )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(f'Test Accuracy Comparison ({dataset_name.upper()})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f"cloning_accuracy_{dataset_name}_{timestamp}.pdf"
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved accuracy comparison to {filepath}")
    
    # 2. Plot metrics comparison (effective rank)
    plt.figure(figsize=(10, 6))
    
    for model_type, history in histories.items():
        # Extract metrics
        epochs = [m[0] for m in history['metrics']]
        
        # Find a common layer for effective rank
        common_layer = None
        for epoch, metrics in history['metrics']:
            for key in metrics:
                if 'effective_rank' in key:
                    common_layer = key.split('/')[0]
                    break
            if common_layer:
                break
        
        if common_layer:
            effective_ranks = [metrics[f"{common_layer}/effective_rank"] 
                            for _, metrics in history['metrics']]
            
            plt.plot(
                epochs,
                effective_ranks,
                label=f"{model_type.capitalize()} Model",
                color=colors[model_type],
                marker='o',
                linewidth=2
            )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Effective Rank', fontsize=12)
    plt.title(f'Effective Rank Comparison ({dataset_name.upper()})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f"cloning_effective_rank_{dataset_name}_{timestamp}.pdf"
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved effective rank comparison to {filepath}")
    
    # 3. Plot weight magnitude comparison
    plt.figure(figsize=(10, 6))
    
    for model_type, history in histories.items():
        # Extract metrics
        epochs = [m[0] for m in history['metrics']]
        
        # Find a common weight parameter
        common_weight = None
        for epoch, metrics in history['metrics']:
            for key in metrics:
                if 'weight' in key and 'l2_norm' in key:
                    common_weight = key
                    break
            if common_weight:
                break
        
        if common_weight:
            weight_norms = [metrics[common_weight] for _, metrics in history['metrics']]
            
            plt.plot(
                epochs,
                weight_norms,
                label=f"{model_type.capitalize()} Model",
                color=colors[model_type],
                marker='o',
                linewidth=2
            )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weight L2 Norm', fontsize=12)
    plt.title(f'Weight Magnitude Comparison ({dataset_name.upper()})', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    filename = f"cloning_weight_norm_{dataset_name}_{timestamp}.pdf"
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved weight norm comparison to {filepath}")
    
    # 4. Plot unexplained variance for cloned model
    if 'cloning_similarity' in histories['cloned']:
        plt.figure(figsize=(10, 6))
        
        epochs = [item['epoch'] for item in histories['cloned']['cloning_similarity']]
        unexplained_vars = [item['unexplained_var'] for item in histories['cloned']['cloning_similarity']]
        
        plt.plot(
            epochs,
            unexplained_vars,
            color='#d62728',  # Red
            marker='o',
            linewidth=2
        )
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Unexplained Variance', fontsize=12)
        plt.title(f'Cloned Model Unexplained Variance ({dataset_name.upper()})', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f"cloning_unexplained_var_{dataset_name}_{timestamp}.pdf"
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved unexplained variance plot to {filepath}")
    
    # 5. Combined metric plot for paper
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Test accuracy (top left)
    for model_type, history in histories.items():
        axes[0, 0].plot(
            range(1, config['epochs'] + 1),
            history['test_acc'],
            label=f"{model_type.capitalize()} Model",
            color=colors[model_type],
            linewidth=2
        )
    
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Test Accuracy', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # Effective rank (top right)
    for model_type, history in histories.items():
        epochs = [m[0] for m in history['metrics']]
        
        # Find a common layer for effective rank
        common_layer = None
        for epoch, metrics in history['metrics']:
            for key in metrics:
                if 'effective_rank' in key:
                    common_layer = key.split('/')[0]
                    break
            if common_layer:
                break
        
        if common_layer:
            effective_ranks = [metrics[f"{common_layer}/effective_rank"] 
                            for _, metrics in history['metrics']]
            
            axes[0, 1].plot(
                epochs,
                effective_ranks,
                label=f"{model_type.capitalize()} Model",
                color=colors[model_type],
                marker='o',
                linewidth=2
            )
    
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Effective Rank', fontsize=12)
    axes[0, 1].set_title('Effective Rank', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Weight norm (bottom left)
    for model_type, history in histories.items():
        epochs = [m[0] for m in history['metrics']]
        
        # Find a common weight parameter
        common_weight = None
        for epoch, metrics in history['metrics']:
            for key in metrics:
                if 'weight' in key and 'l2_norm' in key:
                    common_weight = key
                    break
            if common_weight:
                break
        
        if common_weight:
            weight_norms = [metrics[common_weight] for _, metrics in history['metrics']]
            
            axes[1, 0].plot(
                epochs,
                weight_norms,
                label=f"{model_type.capitalize()} Model",
                color=colors[model_type],
                marker='o',
                linewidth=2
            )
    
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Weight L2 Norm', fontsize=12)
    axes[1, 0].set_title('Weight Magnitude', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # Unexplained variance (bottom right)
    if 'cloning_similarity' in histories['cloned']:
        epochs = [item['epoch'] for item in histories['cloned']['cloning_similarity']]
        unexplained_vars = [item['unexplained_var'] for item in histories['cloned']['cloning_similarity']]
        
        axes[1, 1].plot(
            epochs,
            unexplained_vars,
            color='#d62728',  # Red
            marker='o',
            linewidth=2,
            label='Unexplained Variance'
        )
        
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Unexplained Variance', fontsize=12)
        axes[1, 1].set_title('Cloned Model Unexplained Variance', fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(
        f"Neural Network Cloning Experiment - {dataset_name.upper()}\n"
        f"Base: {config['base_hidden_sizes']}, "
        f"Expanded: {[s*config['expansion_factor'] for s in config['base_hidden_sizes']]}, "
        f"Activation: {config['activation'].upper()}",
        fontsize=16
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the combined figure
    filename = f"cloning_combined_metrics_{dataset_name}_{timestamp}.pdf"
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved combined metrics plot to {filepath}")
    
    return filepath

def main():
    # Define configurations with reduced epochs for testing
    mnist_config = {
        'dataset': 'mnist',
        'base_hidden_sizes': [128, 64],
        'expansion_factor': 2,
        'epochs': 5,  # Reduced for testing
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dropout_p': 0.0,
        'activation': 'relu',
        'normalization': 'batch',
        'metrics_interval': 2  # More frequent metrics collection
    }

    cifar10_config = {
        'dataset': 'cifar10',
        'base_hidden_sizes': [256, 128],
        'expansion_factor': 2,
        'epochs': 5,  # Reduced for testing
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dropout_p': 0.0,
        'activation': 'relu',
        'normalization': 'batch',
        'metrics_interval': 2  # More frequent metrics collection
    }

    # Choose dataset to run based on command line argument or run just MNIST by default
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'mnist'

    if dataset.lower() == 'mnist':
        # Run MNIST experiment only
        mnist_results = run_experiment(mnist_config, dataset_name='mnist')
        mnist_plot_path = plot_experiment_results(mnist_results, 'mnist')
        print(f"MNIST plots saved to: {mnist_plot_path}")
    elif dataset.lower() == 'cifar10':
        # Run CIFAR10 experiment only
        cifar10_results = run_experiment(cifar10_config, dataset_name='cifar10')
        cifar10_plot_path = plot_experiment_results(cifar10_results, 'cifar10')
        print(f"CIFAR10 plots saved to: {cifar10_plot_path}")
    else:
        print(f"Unknown dataset: {dataset}. Please use 'mnist' or 'cifar10'.")

if __name__ == '__main__':
    main()