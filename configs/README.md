# Configuration System

This directory contains default configurations for different models and experiments. The system works with a three-tier approach:

1. **Default Configs**: Base configurations loaded automatically based on model type
2. **Config File Override**: Values from a specified config file override defaults
3. **Command Line Override**: Command line arguments override both defaults and config file

## Default Configuration Files

- `default_experiment.json` - Default experiment parameters
- `default_mlp.json` - Default MLP model architecture configuration
- `default_cnn.json` - Default CNN model architecture configuration
- `default_resnet.json` - Default ResNet model architecture configuration
- `default_vit.json` - Default Vision Transformer architecture configuration

## Example Usage

Run with default parameters for model type and dataset:
```bash
python scripts/run_experiment.py --model mlp --dataset mnist
```

Run with a configuration file:
```bash
python scripts/run_experiment.py --config configs/example_mlp_mnist.json
```

Override config file with command line arguments:
```bash
python scripts/run_experiment.py --config configs/example_mlp_mnist.json --lr 0.005 --epochs 30
```

## Configuration Parameters

Common experiment parameters:
- `dataset`: Dataset to use (mnist, cifar10, cifar100, tiny-imagenet)
- `tasks`: Number of tasks for continual learning
- `classes_per_task`: Number of classes in each task
- `epochs`: Number of epochs to train per task
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `seed`: Random seed for reproducibility
- `no_augment`: Whether to disable data augmentation
- `optimizer`: Optimizer to use (adam, sgd, rmsprop)

Model-specific parameters are available in the respective default config files.