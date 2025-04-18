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
    if model_name.lower() == 'mlp':
        # Parse hidden sizes if provided as string
        hidden_sizes = config.get('hidden_sizes', [512, 256, 128])
        if isinstance(hidden_sizes, str):
            hidden_sizes = [int(x) for x in hidden_sizes.split(',')]
        
        model = MLP(input_size=config.get('input_size', 784),
                  hidden_sizes=hidden_sizes,
                  output_size=config.get('num_classes', 10),
                  activation=config.get('activation', 'relu'),
                  dropout_p=config.get('dropout_p', 0.0),
                  normalization=config.get('normalization', None),
                  norm_after_activation=config.get('norm_after_activation', False),
                  bias=config.get('bias', True),
                  normalization_affine=not config.get('no_affine', False))
    
    elif model_name.lower() == 'cnn':
        model = CNN(in_channels=config.get('in_channels', 3),
                  conv_channels=config.get('conv_channels', [64, 128, 256]),
                  kernel_sizes=config.get('kernel_sizes', [3, 3, 3]),
                  strides=config.get('strides', [1, 1, 1]),
                  paddings=config.get('paddings', [1, 1, 1]),
                  fc_hidden_units=config.get('fc_hidden_units', [512]),
                  num_classes=config.get('num_classes', 10),
                  input_size=config.get('input_size', 32),
                  activation=config.get('activation', 'relu'),
                  dropout_p=config.get('dropout_p', 0.0),
                  pool_type=config.get('pool_type', 'max'),
                  pool_size=config.get('pool_size', 2),
                  use_batchnorm=config.get('normalization', 'bn').lower() == 'bn',
                  norm_after_activation=config.get('norm_after_activation', False),
                  normalization_affine=not config.get('no_affine', False))
    
    elif model_name.lower() == 'resnet':
        model = ResNet(layers=config.get('layers', [2, 2, 2, 2]),
                     num_classes=config.get('num_classes', 10),
                     in_channels=config.get('in_channels', 3),
                     base_channels=config.get('base_channels', 64),
                     activation=config.get('activation', 'relu'),
                     dropout_p=config.get('dropout_p', 0.0),
                     use_batchnorm=config.get('normalization', 'bn').lower() == 'bn',
                     norm_after_activation=config.get('norm_after_activation', False),
                     normalization_affine=not config.get('no_affine', False))
    
    elif model_name.lower() == 'vit':
        model = VisionTransformer(img_size=config.get('img_size', 32),
                                patch_size=config.get('patch_size', 4),
                                in_channels=config.get('in_channels', 3),
                                num_classes=config.get('num_classes', 10),
                                embed_dim=config.get('embed_dim', 192),
                                depth=config.get('depth', 12),
                                n_heads=config.get('n_heads', 8),
                                mlp_ratio=config.get('mlp_ratio', 4.0),
                                qkv_bias=config.get('qkv_bias', True),
                                drop_rate=config.get('drop_rate', 0.1),
                                attn_drop_rate=config.get('attn_drop_rate', 0.0),
                                activation=config.get('activation', 'gelu'),
                                normalization=config.get('normalization', 'layer'),
                                normalization_affine=not config.get('no_affine', False))
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def run_experiment(args):
    """
    Run a neural network experiment with the specified arguments.
    
    Args:
        args: Namespace containing experiment parameters
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create transforms with or without augmentation
    transform_train, transform_test = get_transforms(args.dataset, args.no_augment)
    
    # Get dataset
    train_dataset, test_dataset, num_classes = get_dataset(
        args.dataset, transform_train, transform_test)
    
    # Parse partitions if provided
    if hasattr(args, 'partitions') and args.partitions is not None:
        try:
            partitions = eval(args.partitions)
            print(f"Using custom partitions: {partitions}")
        except:
            print(f"Error parsing partitions: {args.partitions}")
            partitions = None
    else:
        partitions = None
    
    # Create task sequence by splitting classes
    if partitions is None:
        # Auto-generate partitions
        if args.tasks is None:
            args.tasks = num_classes // args.classes_per_task
        classes_per_task = args.classes_per_task if hasattr(args, 'classes_per_task') else num_classes // args.tasks
        class_sequence = [
            list(range(i * classes_per_task, min((i + 1) * classes_per_task, num_classes)))
            for i in range(args.tasks)
        ]
    else:
        # Use provided partitions
        class_sequence = [list(p) for p in partitions]
        args.tasks = len(class_sequence)
    
    print(f"Class sequence: {class_sequence}")
    
    # Handle hidden sizes
    if hasattr(args, 'hidden_sizes') and args.hidden_sizes:
        if isinstance(args.hidden_sizes, str):
            hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
        else:
            hidden_sizes = args.hidden_sizes
    else:
        hidden_sizes = [512, 256, 128]
    
    # Create class partitions for continual learning
    partitioned_datasets = create_class_partitions(
        train_dataset, [tuple(cls_list) for cls_list in class_sequence])
    
    # Create data loaders for each partition
    task_dataloaders = {}
    for task_id, task_dataset in enumerate(partitioned_datasets):
        # Split into training and validation
        dataset_size = len(task_dataset)
        val_split = 0.2  # Using 20% for validation
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_subset = Subset(task_dataset, train_indices)
        val_subset = Subset(task_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Fixed batches for metrics
        fixed_train = Subset(train_subset, range(min(500, len(train_subset))))
        fixed_val = Subset(val_subset, range(min(500, len(val_subset))))
        
        fixed_train_loader = DataLoader(fixed_train, batch_size=args.batch_size, shuffle=False)
        fixed_val_loader = DataLoader(fixed_val, batch_size=args.batch_size, shuffle=False)
        
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
    
    # Determine input size based on dataset
    if args.dataset.lower() == 'mnist':
        input_size = 28 * 28
        in_channels = 1
        img_size = 28
    elif args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
        input_size = 32 * 32 * 3
        in_channels = 3
        img_size = 32
    elif args.dataset.lower() == 'tiny-imagenet':
        input_size = 64 * 64 * 3
        in_channels = 3
        img_size = 64
    else:
        input_size = 32 * 32 * 3
        in_channels = 3
        img_size = 32
    
    # Model configuration
    model_config = {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_size": input_size if args.model.lower() == 'mlp' else img_size,
        "hidden_sizes": hidden_sizes,
        "activation": args.activation if hasattr(args, 'activation') else 'relu',
        "dropout_p": args.dropout if hasattr(args, 'dropout') else 0.0,
        "normalization": args.normalization,
        "norm_after_activation": args.norm_after_activation if hasattr(args, 'norm_after_activation') else False,
        "no_affine": args.no_affine if hasattr(args, 'no_affine') else False,
    }
    
    # Create model
    model = get_model(args.model, model_config).to(device)
    print(f"Created {args.model.upper()} model")
    
    # Additional configuration for training
    config = {
        "learning_rate": args.lr,
        "epochs_per_task": args.epochs,
        "metrics_frequency": 5,
        "dead_threshold": 0.95,
        "corr_threshold": 0.95,
        "saturation_threshold": 1e-4,
        "saturation_percentage": 0.99,
        "optimizer": args.optimizer if hasattr(args, 'optimizer') else 'adam',
        "reinit_output": args.reinit_output if hasattr(args, 'reinit_output') else False,
        "reinit_adam": args.reinit_adam if hasattr(args, 'reinit_adam') else False,
        "reset": args.reset if hasattr(args, 'reset') else False,
        "early_stopping_steps": args.early_stopping_steps if hasattr(args, 'early_stopping_steps') else 0,
        "model_type": args.model
    }
    
    # Initialize W&B if requested and available
    if not hasattr(args, 'no_wandb') or not args.no_wandb:
        if WANDB_AVAILABLE:
            wandb_project = args.wandb_project if hasattr(args, 'wandb_project') else "continual-learning-experiment"
            wandb_entity = args.wandb_entity if hasattr(args, 'wandb_entity') else None
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                config={
                    "model": args.model,
                    "dataset": args.dataset,
                    "n_tasks": args.tasks,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "epochs_per_task": args.epochs,
                    "seed": args.seed,
                    "activation": model_config["activation"],
                    "dropout": model_config["dropout_p"],
                    "normalization": model_config["normalization"],
                    "norm_after_activation": model_config["norm_after_activation"],
                    "no_augment": args.no_augment if hasattr(args, 'no_augment') else False,
                    **config
                }
            )
        else:
            print("Warning: Weights & Biases (wandb) not installed. Running without wandb logging.")
    
    # Check for dryrun
    if hasattr(args, 'dryrun') and args.dryrun:
        print("Dry run completed, exiting without training.")
        return None
    
    # Train using continual learning
    history = train_continual_learning(
        model, task_dataloaders, config, device=device)
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'saved_models/{args.model}_{args.dataset}_{args.tasks}tasks.pth')
    
    # Finish W&B run
    if not hasattr(args, 'no_wandb') or not args.no_wandb:
        if WANDB_AVAILABLE:
            wandb.finish()
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Run a Neural Network Dynamic Scaling experiment")
    
    # Config file option
    parser.add_argument("--config", help="Path to a JSON configuration file")
    
    # Common experiment parameters
    parser.add_argument("--model", type=str, default='cnn', choices=['mlp', 'cnn', 'resnet', 'vit'],
                      help='Model architecture')
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet'],
                      help='Dataset to use')

    parser.add_argument("--epochs", type=int, default=20,
                      help='Number of epochs to train each task')
    parser.add_argument("--batch-size", type=int, default=128,
                      help='Batch size for training')
    parser.add_argument("--lr", type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument("--seed", type=int, default=42,
                      help='Random seed')
    parser.add_argument("--device", type=str, default=None,
                      help='Device to use for training (cpu, cuda, mps). If not specified, best available device will be used.')
    parser.add_argument("--no-augment", action="store_true",
                      help='Disable data augmentation')
    parser.add_argument("--tasks", type=int, default=None,
                      help='Number of tasks (each task contains new classes)')
    parser.add_argument("--classes-per-task", type=int, default=2,
                      help='Optional: Specify exact number of classes per task. ')
    parser.add_argument("--partitions", type=str, default=None,
                      help='Task partitions, as a Python list of tuples. Example: "[(0,1), (2,3), (4,5), (6,7,8,9)]"')
    
    # Model parameters
    parser.add_argument("--hidden-sizes", type=str, default=None,
                      help='Comma-separated list of hidden layer sizes (e.g., "512,512,256")')
    parser.add_argument("--dropout", type=float, default=0.0,
                      help='Dropout probability')
    parser.add_argument("--activation", type=str, default='relu', 
                      choices=['relu', 'gelu', 'silu', 'elu', 'tanh', 'sigmoid', 'mish'],
                      help='Type of activation function to use in the model')
    parser.add_argument("--normalization", type=str, default='none', 
                      choices=['none', 'batch', 'layer'],
                      help='Type of normalization to use (none, bn=batch norm, ln=layer norm)')
    parser.add_argument("--norm-after-activation", action="store_true",
                      help='Apply normalization after activation instead of before')
    parser.add_argument("--no-affine", action="store_true",
                      help='Disable affine parameters (learnable scale and shift) in normalization layers')
    
    # Optimization parameters
    parser.add_argument("--optimizer", type=str, default='adam',
                      choices=['adam', 'sgd', 'rmsprop'],
                      help='Optimizer to use for training')
    parser.add_argument("--reset", action="store_true",
                      help='Reset model weights before training on each new task')
    parser.add_argument("--reinit-output", action="store_true",
                      help='Reinitialize output weights for task classes at the beginning of each task')
    parser.add_argument("--reinit-adam", action="store_true",
                      help='Reinitialize optimizer state (momentum and variance) for each new task')
    parser.add_argument("--early-stopping-steps", type=int, default=0,
                      help='Number of epochs for early stopping patience. 0 disables early stopping.')
    parser.add_argument("--summary", action="store_true",
                      help='Show summary of all task accuracies after each task completes')
    parser.add_argument("--dryrun", action="store_true",
                      help='Only setup the partitions and exit without training')
    
    # Weights & Biases logging
    parser.add_argument("--use-wandb", action="store_true",
                      help='Use Weights & Biases for experiment tracking')
    parser.add_argument("--no-wandb", action="store_true",
                      help='Disable Weights & Biases logging')
    parser.add_argument("--wandb-project", type=str, default='continual-learning-experiment',
                      help='Weights & Biases project name')
    parser.add_argument("--wandb-entity", type=str, default=None,
                      help='Weights & Biases entity name (username or team name)')
    
    # Parse args
    args = parser.parse_args()
    
    # If config file is provided, update args with config values
    if args.config:
        config = parse_config_file(args.config)
        args_dict = vars(args)
        # Only update args that weren't explicitly set on the command line
        # and are present in the config file
        for k, v in config.items():
            if k in args_dict and args_dict[k] is None:
                setattr(args, k, v)
    
    # Run the experiment
    run_experiment(args)
    return 0

if __name__ == "__main__":
    sys.exit(main())