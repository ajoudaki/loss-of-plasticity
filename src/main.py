import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import os
import wandb

from .models import MLP, CNN, ResNet, VisionTransformer
from .utils.data import prepare_continual_learning_data
from .training.continual_learn import train_continual_learning

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataset(dataset_name, download=True):
    """Get the specified dataset."""
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=download, transform=transform)
            
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=download, transform=transform)
        
        num_classes = 10
            
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=download, transform=transform)
            
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=download, transform=transform)
        
        num_classes = 10
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes

def get_model(model_name, config):
    """Create a neural network model based on the specified name and configuration."""
    if model_name.lower() == 'mlp':
        model = MLP(input_size=config.get('input_size', 784),
                  hidden_sizes=config.get('hidden_sizes', [512, 256, 128]),
                  output_size=config.get('num_classes', 10),
                  activation=config.get('activation', 'relu'),
                  dropout_p=config.get('dropout_p', 0.0),
                  normalization=config.get('normalization', None),
                  norm_after_activation=config.get('norm_after_activation', False),
                  bias=config.get('bias', True),
                  normalization_affine=config.get('normalization_affine', True))
    
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
                  use_batchnorm=config.get('use_batchnorm', True),
                  norm_after_activation=config.get('norm_after_activation', False),
                  normalization_affine=config.get('normalization_affine', True))
    
    elif model_name.lower() == 'resnet':
        model = ResNet(layers=config.get('layers', [2, 2, 2, 2]),
                     num_classes=config.get('num_classes', 10),
                     in_channels=config.get('in_channels', 3),
                     base_channels=config.get('base_channels', 64),
                     activation=config.get('activation', 'relu'),
                     dropout_p=config.get('dropout_p', 0.0),
                     use_batchnorm=config.get('use_batchnorm', True),
                     norm_after_activation=config.get('norm_after_activation', False),
                     normalization_affine=config.get('normalization_affine', True))
    
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
                                normalization_affine=config.get('normalization_affine', True))
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Continual Learning Experiment')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn', 'resnet', 'vit'],
                      help='Model architecture')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                      help='Dataset to use')
    parser.add_argument('--tasks', type=int, default=5,
                      help='Number of tasks (each task contains new classes)')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs to train each task')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Additional configuration
    config = {
        "learning_rate": args.lr,
        "epochs_per_task": args.epochs,
        "metrics_frequency": 5,
        "dead_threshold": 0.95,
        "corr_threshold": 0.95,
        "saturation_threshold": 1e-4,
        "saturation_percentage": 0.99
    }
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project="continual-learning-experiment",
            config={
                "model": args.model,
                "dataset": args.dataset,
                "n_tasks": args.tasks,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "epochs_per_task": args.epochs,
                "seed": args.seed,
                **config
            }
        )
    
    # Get dataset
    train_dataset, test_dataset, num_classes = get_dataset(args.dataset)
    
    # Create task sequence by splitting classes
    classes_per_task = num_classes // args.tasks
    class_sequence = [
        list(range(i * classes_per_task, (i + 1) * classes_per_task))
        for i in range(args.tasks)
    ]
    
    print(f"Class sequence: {class_sequence}")
    
    # Prepare data loaders for continual learning
    task_dataloaders = prepare_continual_learning_data(
        train_dataset, class_sequence, batch_size=args.batch_size)
    
    # Model configuration
    model_config = {
        "num_classes": num_classes,
        "in_channels": 3 if args.dataset == 'cifar10' else 1,
        "input_size": 32 if args.dataset == 'cifar10' else 28,
    }
    
    # Create model
    model = get_model(args.model, model_config).to(args.device)
    print(f"Created {args.model.upper()} model")
    
    # Train using continual learning
    history = train_continual_learning(
        model, task_dataloaders, config, device=args.device)
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    torch.save(model.state_dict(), f'saved_models/{args.model}_{args.dataset}_{args.tasks}tasks.pth')
    
    # Finish W&B run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()