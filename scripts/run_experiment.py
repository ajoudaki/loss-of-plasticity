#!/usr/bin/env python3
"""
Script to run experiments with the Neural Network Dynamic Scaling codebase.
This combines the functionality of the main entry point and experiment runner
into a single script for ease of use.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
from src.models import MLP, CNN, ResNet, VisionTransformer
from src.utils.data import (
    get_transforms, 
    get_dataset, 
    create_class_partitions
)
from src.utils.layers import set_seed
from src.training.train_continual import train_continual_learning

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def get_device(device_str=None):
    """
    Get the appropriate torch device.
    
    Args:
        device_str: Device string (e.g., 'cuda', 'cpu', 'mps')
                   If None, will select the best available device
    
    Returns:
        torch.device: The selected device
    """
    if device_str is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_str)

def parse_config_file(config_path):
    """
    Parse a JSON configuration file for experiment settings.
    
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)

def load_default_config(config_type):
    """
    Load a default configuration file for the specified type.
    
    Args:
        config_type: Type of configuration to load ('experiment', 'mlp', 'cnn', 'resnet', 'vit')
        
    Returns:
        dict: Default configuration parameters
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'default_{config_type}.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load default {config_type} config: {e}")
        return {}

def reinitialize_output_weights(model, task_classes, model_type='mlp'):
    """
    Reinitialize the output weights for the specified task classes.
    
    Args:
        model: The neural network model
        task_classes: List of class indices for the current task
        model_type: Type of model ('mlp', 'cnn', 'resnet', or 'vit')
    """
    # Get the output layer
    if model_type == 'mlp':
        # For MLP, the output layer is the last layer in the sequential model
        output_layer = model.model[-1]
    elif model_type == 'cnn':
        # For CNN, the output layer is typically the last fully connected layer
        output_layer = model.fc_layers[-1]
    elif model_type == 'resnet':
        # For ResNet, the output layer is the linear layer
        output_layer = model.fc
    elif model_type == 'vit':
        # For ViT, the output layer is the head
        output_layer = model.head
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Only reinitialize weights for task classes
    with torch.no_grad():
        # Reinitialize weights for task classes only
        layer_norm = (output_layer.weight**2).mean().item()**0.5
        for cls in range(len(output_layer.weight)):
            # Initialize the weights for this class, use layer std to 
            nn.init.normal_(output_layer.weight[cls], std=layer_norm)
            # Initialize the bias for this class
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias[cls])
    
    print(f"Reinitialized output weights for classes: {task_classes}")

def get_model(model_name, config):
    """
    Create a neural network model based on the specified name and configuration.
    
    Args:
        model_name: Name of the model architecture
        config: Dictionary with model configuration parameters
        
    Returns:
        model: The initialized PyTorch model
    """
    model_name = model_name.lower()
    
    # Parse hidden sizes if provided as string
    if 'hidden_sizes' in config and isinstance(config['hidden_sizes'], str):
        config['hidden_sizes'] = [int(x) for x in config['hidden_sizes'].split(',')]
    
    if model_name == 'mlp':
        model = MLP(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['num_classes'],
            activation=config['activation'],
            dropout_p=config['dropout_p'],
            normalization=config['normalization'],
            norm_after_activation=config['norm_after_activation'],
            bias=config['bias'],
            normalization_affine=config['normalization_affine']
        )
    
    elif model_name == 'cnn':
        model = CNN(
            in_channels=config['in_channels'],
            conv_channels=config['conv_channels'],
            kernel_sizes=config['kernel_sizes'],
            strides=config['strides'],
            paddings=config['paddings'],
            fc_hidden_units=config['fc_hidden_units'],
            num_classes=config['num_classes'],
            input_size=config['input_size'],
            activation=config['activation'],
            dropout_p=config['dropout_p'],
            pool_type=config['pool_type'],
            pool_size=config['pool_size'],
            use_batchnorm=config['use_batchnorm'],
            norm_after_activation=config['norm_after_activation'],
            normalization_affine=config['normalization_affine']
        )
    
    elif model_name == 'resnet':
        model = ResNet(
            layers=config['layers'],
            num_classes=config['num_classes'],
            in_channels=config['in_channels'],
            base_channels=config['base_channels'],
            activation=config['activation'],
            dropout_p=config['dropout_p'],
            use_batchnorm=config['use_batchnorm'],
            norm_after_activation=config['norm_after_activation'],
            normalization_affine=config['normalization_affine']
        )
    
    elif model_name == 'vit':
        model = VisionTransformer(
            img_size=config['input_size'],
            patch_size=config['patch_size'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            n_heads=config['n_heads'],
            mlp_ratio=config['mlp_ratio'],
            qkv_bias=config['qkv_bias'],
            drop_rate=config['drop_rate'],
            attn_drop_rate=config['attn_drop_rate'],
            activation=config['activation'],
            normalization=config['normalization'],
            normalization_affine=config['normalization_affine']
        )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def run_experiment(args):
    """
    Run a neural network experiment with the specified arguments.
    
    Args:
        args: Namespace containing experiment parameters
    """
    # Load default experiment configuration
    exp_config = load_default_config('experiment')
    
    # Convert args to dictionary for easier manipulation
    args_dict = vars(args)
    
    # Define dataset-specific parameters based on the dataset
    dataset = args_dict.get('dataset', exp_config.get('dataset', 'cifar10')).lower()
    
    if dataset == 'mnist':
        dataset_params = {
            'input_size': 28 * 28,  # For MLP
            'img_size': 28,         # For other models
            'in_channels': 1,
            'num_classes': 10
        }
    elif dataset == 'cifar10':
        dataset_params = {
            'input_size': 32 * 32 * 3,
            'img_size': 32,
            'in_channels': 3,
            'num_classes': 10
        }
    elif dataset == 'cifar100':
        dataset_params = {
            'input_size': 32 * 32 * 3,
            'img_size': 32,
            'in_channels': 3,
            'num_classes': 100
        }
    elif dataset == 'tiny-imagenet':
        dataset_params = {
            'input_size': 64 * 64 * 3,
            'img_size': 64,
            'in_channels': 3,
            'num_classes': 200
        }
    else:
        dataset_params = {
            'input_size': 32 * 32 * 3,
            'img_size': 32,
            'in_channels': 3,
            'num_classes': 10
        }
        print(f"Warning: Unknown dataset '{dataset}'. Using default parameters.")
    
    # Set random seed for reproducibility
    seed = args_dict.get('seed', exp_config.get('seed', 42))
    set_seed(seed)
    
    # Get device
    device = get_device(args_dict.get('device'))
    print(f"Using device: {device}")
    
    # Determine model type
    model_type = args_dict.get('model', 'cnn').lower()
    
    # Load default model configuration
    model_config = load_default_config(model_type)
    
    # Create transforms with or without augmentation
    no_augment = args_dict.get('no_augment', exp_config.get('no_augment', False))
    transform_train, transform_test = get_transforms(dataset, no_augment)
    
    # Get dataset
    train_dataset, val_dataset, num_classes = get_dataset(dataset, transform_train, transform_test)
    dataset_params['num_classes'] = num_classes  # Update with actual number from dataset
    
    # Parse partitions if provided
    partitions = None
    if 'partitions' in args_dict and args_dict['partitions'] is not None:
        try:
            partitions = eval(args_dict['partitions'])
            print(f"Using custom partitions: {partitions}")
        except:
            print(f"Error parsing partitions: {args_dict['partitions']}")
    
    # Get task configuration
    tasks = args_dict.get('tasks', exp_config.get('tasks'))
    classes_per_task = args_dict.get('classes_per_task', exp_config.get('classes_per_task', 2))
    
    # Create task sequence by splitting classes
    if partitions is None:
        # Auto-generate partitions
        if tasks is None:
            tasks = num_classes // classes_per_task
        
        class_sequence = [
            list(range(i * classes_per_task, min((i + 1) * classes_per_task, num_classes)))
            for i in range(tasks)
        ]
    else:
        # Use provided partitions
        class_sequence = [list(p) for p in partitions]
        tasks = len(class_sequence)
    
    print(f"Class sequence: {class_sequence}")
    
    # Create class partitions for continual learning
    partitioned_train_datasets = create_class_partitions(
        train_dataset, [tuple(cls_list) for cls_list in class_sequence])
    
    partitioned_val_datasets = create_class_partitions(
        val_dataset, [tuple(cls_list) for cls_list in class_sequence])
    
    # Create data loaders for each partition
    batch_size = args_dict.get('batch_size', exp_config.get('batch_size', 128))
    task_dataloaders = {}
    
    for task_id, (train_subset, val_subset) in enumerate(zip(partitioned_train_datasets, partitioned_val_datasets)):
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=batch_size, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=batch_size, shuffle=False)
        
        task_dataloaders[task_id] = {
            'current': {
                'train': train_loader,
                'val': val_loader,
                'fixed_train': fixed_train_loader,
                'fixed_val': fixed_val_loader,
                'classes': class_sequence[task_id]
            },
            'old': {}
        }
    
    # Create final model configuration by merging:
    # 1. Default model configuration
    # 2. Dataset-specific parameters
    # 3. Command-line arguments
    
    # Adjust input size based on model type
    if model_type == 'mlp':
        dataset_params['input_size'] = dataset_params['input_size']  # Use flattened image
    else:
        dataset_params['input_size'] = dataset_params['img_size']    # Use image size
    
    # Start with default model config
    final_model_config = model_config.copy()
    
    # Add dataset-specific parameters (overrides defaults)
    final_model_config.update(dataset_params)
    
    # Extract any model-specific parameters from args and override both defaults and dataset params
    for key in model_config.keys():
        if key in args_dict and args_dict[key] is not None:
            final_model_config[key] = args_dict[key]
    
    # Create model
    model = get_model(model_type, final_model_config).to(device)
    print(f"Created {model_type.upper()} model")
    
    # Training configuration - start with defaults
    train_config = {
        "learning_rate": exp_config.get('lr', 0.001),
        "epochs_per_task": exp_config.get('epochs', 20),
        "metrics_frequency": exp_config.get('metrics_frequency', 5),
        "dead_threshold": exp_config.get('dead_threshold', 0.95),
        "corr_threshold": exp_config.get('corr_threshold', 0.95),
        "saturation_threshold": exp_config.get('saturation_threshold', 1e-4),
        "saturation_percentage": exp_config.get('saturation_percentage', 0.99),
        "optimizer": exp_config.get('optimizer', 'adam'),
        "reinit_output": exp_config.get('reinit_output', False),
        "reinit_adam": exp_config.get('reinit_adam', False),
        "reset": exp_config.get('reset', False),
        "early_stopping_steps": exp_config.get('early_stopping_steps', 0),
        "model_type": model_type
    }
    
    # Override with command-line arguments
    for key in train_config.keys():
        if key in args_dict and args_dict[key] is not None:
            train_config[key] = args_dict[key]
    
    # Special case for lr since the config uses 'lr' but train_config uses 'learning_rate'
    if 'lr' in args_dict and args_dict['lr'] is not None:
        train_config['learning_rate'] = args_dict['lr']
    
    # Initialize W&B if requested and available
    use_wandb = not args_dict.get('no_wandb', exp_config.get('no_wandb', True))
    if use_wandb:
        if WANDB_AVAILABLE:
            wandb_project = args_dict.get('wandb_project', "continual-learning-experiment")
            wandb_entity = args_dict.get('wandb_entity')
            
            # Prepare wandb config
            wandb_config = {
                "model": model_type,
                "dataset": dataset,
                "n_tasks": tasks,
                "batch_size": batch_size,
                "lr": train_config["learning_rate"],
                "epochs_per_task": train_config["epochs_per_task"],
                "seed": seed,
                "no_augment": no_augment,
            }
            
            # Add model config
            for k, v in final_model_config.items():
                if k not in ["num_classes", "in_channels", "input_size"]:
                    wandb_config[f"model_{k}"] = v
            
            # Add training config
            for k, v in train_config.items():
                if k not in wandb_config:
                    wandb_config[k] = v
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config
            )
        else:
            print("Warning: Weights & Biases (wandb) not installed. Running without wandb logging.")
    
    # Check for dryrun
    if args_dict.get('dryrun', False):
        print("Dry run completed, exiting without training.")
        return None
    
    # Train using continual learning
    history = train_continual_learning(
        model, task_dataloaders, train_config, device=device)
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'saved_models/{model_type}_{dataset}_{tasks}tasks.pth')
    
    # Finish W&B run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Run a Neural Network Dynamic Scaling experiment")
    
    # Config file option
    parser.add_argument("--config", help="Path to a JSON configuration file")
    
    # Common experiment parameters
    parser.add_argument("--model", type=str, choices=['mlp', 'cnn', 'resnet', 'vit'],
                      help='Model architecture')
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet'],
                      help='Dataset to use')

    parser.add_argument("--epochs", type=int,
                      help='Number of epochs to train each task')
    parser.add_argument("--batch-size", type=int,
                      help='Batch size for training')
    parser.add_argument("--lr", type=float,
                      help='Learning rate')
    parser.add_argument("--seed", type=int,
                      help='Random seed')
    parser.add_argument("--device", type=str,
                      help='Device to use for training (cpu, cuda, mps). If not specified, best available device will be used.')
    parser.add_argument("--no-augment", action="store_true",
                      help='Disable data augmentation')
    parser.add_argument("--tasks", type=int,
                      help='Number of tasks (each task contains new classes)')
    parser.add_argument("--classes-per-task", type=int,
                      help='Specify exact number of classes per task')
    parser.add_argument("--partitions", type=str,
                      help='Task partitions, as a Python list of tuples. Example: "[(0,1), (2,3), (4,5), (6,7,8,9)]"')
    
    # Model parameters - common for all models
    parser.add_argument("--dropout-p", type=float,
                      help='Dropout probability')
    parser.add_argument("--activation", type=str, 
                      choices=['relu', 'gelu', 'silu', 'elu', 'tanh', 'sigmoid', 'mish'],
                      help='Type of activation function to use in the model')
    parser.add_argument("--normalization", type=str, 
                      choices=['none', 'batch', 'layer'],
                      help='Type of normalization to use')
    parser.add_argument("--norm-after-activation", action="store_true",
                      help='Apply normalization after activation instead of before')
    parser.add_argument("--normalization-affine", action="store_true",
                      help='Enable affine parameters in normalization layers')
    
    # MLP-specific parameters
    parser.add_argument("--hidden-sizes", type=str,
                      help='Comma-separated list of hidden layer sizes (e.g., "512,512,256")')
    parser.add_argument("--bias", action="store_true",
                      help='Use bias in linear layers (for MLP)')
    
    # CNN-specific parameters
    parser.add_argument("--conv-channels", type=str,
                      help='Comma-separated list of conv channels (e.g., "64,128,256")')
    parser.add_argument("--kernel-sizes", type=str,
                      help='Comma-separated list of kernel sizes (e.g., "3,3,3")')
    parser.add_argument("--strides", type=str,
                      help='Comma-separated list of stride values (e.g., "1,1,1")')
    parser.add_argument("--paddings", type=str,
                      help='Comma-separated list of padding values (e.g., "1,1,1")')
    parser.add_argument("--fc-hidden-units", type=str,
                      help='Comma-separated list of FC layer sizes (e.g., "512,256")')
    parser.add_argument("--pool-type", type=str, choices=['max', 'avg'],
                      help='Type of pooling to use')
    parser.add_argument("--pool-size", type=int,
                      help='Pooling kernel size')
    parser.add_argument("--use-batchnorm", action="store_true",
                      help='Use batch normalization in CNN')
    
    # ResNet-specific parameters
    parser.add_argument("--layers", type=str,
                      help='Comma-separated list of layer counts for ResNet (e.g., "2,2,2,2")')
    parser.add_argument("--base-channels", type=int,
                      help='Base channel count for ResNet')
    
    # ViT-specific parameters
    parser.add_argument("--patch-size", type=int,
                      help='Patch size for Vision Transformer')
    parser.add_argument("--embed-dim", type=int,
                      help='Embedding dimension for Vision Transformer')
    parser.add_argument("--depth", type=int,
                      help='Number of transformer blocks for Vision Transformer')
    parser.add_argument("--n-heads", type=int,
                      help='Number of attention heads for Vision Transformer')
    parser.add_argument("--mlp-ratio", type=float,
                      help='MLP ratio for Vision Transformer')
    parser.add_argument("--qkv-bias", action="store_true",
                      help='Use bias in QKV projections for Vision Transformer')
    parser.add_argument("--drop-rate", type=float,
                      help='Dropout rate for Vision Transformer')
    parser.add_argument("--attn-drop-rate", type=float,
                      help='Attention dropout rate for Vision Transformer')
    
    # Optimization parameters
    parser.add_argument("--optimizer", type=str, choices=['adam', 'sgd', 'rmsprop'],
                      help='Optimizer to use for training')
    parser.add_argument("--reset", action="store_true",
                      help='Reset model weights before training on each new task')
    parser.add_argument("--reinit-output", action="store_true",
                      help='Reinitialize output weights for task classes at the beginning of each task')
    parser.add_argument("--reinit-adam", action="store_true",
                      help='Reinitialize optimizer state (momentum and variance) for each new task')
    parser.add_argument("--early-stopping-steps", type=int,
                      help='Number of epochs for early stopping patience. 0 disables early stopping.')
    parser.add_argument("--metrics-frequency", type=int,
                      help='Frequency (in epochs) to collect network metrics')
    parser.add_argument("--summary", action="store_true",
                      help='Show summary of all task accuracies after each task completes')
    parser.add_argument("--dryrun", action="store_true",
                      help='Only setup the partitions and exit without training')
    
    # Weights & Biases logging
    parser.add_argument("--use-wandb", action="store_true",
                      help='Use Weights & Biases for experiment tracking')
    parser.add_argument("--no-wandb", action="store_true",
                      help='Disable Weights & Biases logging')
    parser.add_argument("--wandb-project", type=str,
                      help='Weights & Biases project name')
    parser.add_argument("--wandb-entity", type=str,
                      help='Weights & Biases entity name (username or team name)')
    
    # Parse args
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Load default experiment config
    experiment_config = load_default_config('experiment')
    
    # Load config from file if provided
    file_config = {}
    if args.config:
        file_config = parse_config_file(args.config)
    
    # Convert comma-separated string arguments to lists
    for arg_name in ["hidden_sizes", "conv_channels", "kernel_sizes", "strides", 
                    "paddings", "fc_hidden_units", "layers"]:
        if arg_name in args_dict and args_dict[arg_name] is not None:
            try:
                args_dict[arg_name] = [int(x) for x in args_dict[arg_name].split(',')]
            except ValueError:
                print(f"Error parsing {arg_name}. Expected comma-separated integers.")
                sys.exit(1)
    
    # Now run the experiment with properly configured args
    run_experiment(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())