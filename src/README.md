# Neural Network Dynamic Scaling Project

This project has been refactored to follow a modular structure for better organization and maintainability.

## Project Structure

```
project/
├── models/                  # Model definitions
│   ├── __init__.py          # Exports all models
│   ├── mlp.py               # MLP model
│   ├── cnn.py               # CNN model
│   ├── resnet.py            # ResNet model
│   └── vit.py               # Vision Transformer model
├── utils/
│   ├── __init__.py
│   ├── layers.py            # Layer utilities (get_activation, get_normalization)
│   ├── metrics.py           # Metrics calculation functions
│   ├── monitor.py           # NetworkMonitor class
│   ├── data.py              # Data loading utilities
│   └── visualization.py     # Plotting functions
└── training/
    ├── __init__.py
    ├── eval.py              # Evaluation functions
    └── train_continual.py   # Continual learning training loop
```

## Components

### Models

- **MLP**: Multi-layer perceptron with customizable architecture
- **CNN**: Convolutional Neural Network with configurable layers
- **ResNet**: Residual Network implementation for continual learning
- **VisionTransformer**: Implementation of ViT with customizable parameters

### Utils

- **layers.py**: Common layer utilities including activation and normalization functions
- **metrics.py**: Functions for measuring network properties (dead neurons, saturation, etc.)
- **monitor.py**: NetworkMonitor class for tracking activations and gradients
- **data.py**: Data preparation utilities for continual learning
- **visualization.py**: Plotting and visualization tools

### Training

- **eval.py**: Evaluation functions for model assessment
- **train_continual.py**: Main continual learning training loop implementation

## Scripts

The project includes several utility scripts in the `scripts/` directory:

- **run_experiment.py**: Main entry point for running experiments (replaces main.py)
- **download_tiny_imagenet.py**: Download and extract the Tiny ImageNet dataset
- **extract_notebook.py**: Extract code from Jupyter notebooks
- **check_imports.py**: Check for missing imports in the codebase

## Key Features

- Modular architecture with clear separation of concerns
- Comprehensive metrics tracking during training
- Support for different model architectures
- Configurable hyperparameters
- Wandb integration for experiment tracking
- Step and epoch tracking at both task and global levels

## Usage

The main entry point is now the `run_experiment.py` script in the scripts directory:

```
python scripts/run_experiment.py --model cnn --dataset cifar10 --tasks 5 --epochs 20 --batch-size 128 --lr 0.001 --seed 42 --use-wandb
```

You can also use JSON configuration files:

```
python scripts/run_experiment.py --config configs/default_experiment.json
```

## Metrics Tracked

During training, the system tracks:
- Basic metrics: loss, accuracy
- Network health metrics: dead neurons, duplicated neurons, effective rank, stable rank, neuron saturation
- Global and task-specific progress tracking