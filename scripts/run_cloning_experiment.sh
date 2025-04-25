#!/bin/bash
# Example script for running neural network cloning experiments

# Exit on error
set -e

# Display help information
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Neural Network Cloning Experiment Runner"
  echo ""
  echo "Usage: ./run_cloning_experiment.sh [OPTION]"
  echo ""
  echo "Options:"
  echo "  mlp-mnist     Run MLP cloning experiment on MNIST"
  echo "  cnn-cifar10   Run CNN cloning experiment on CIFAR-10"
  echo "  resnet-cifar10 Run ResNet cloning experiment on CIFAR-10"
  echo "  vit-cifar10   Run ViT cloning experiment on CIFAR-10"
  echo "  custom        Run with custom parameters (edit script to modify)"
  echo "  -h, --help    Display this help message"
  echo ""
  echo "Example: ./run_cloning_experiment.sh mlp-mnist"
  exit 0
fi

# Default option if none provided
OPTION=${1:-mlp-mnist}

echo "Running cloning experiment: $OPTION"

case $OPTION in
  mlp-mnist)
    # Basic MLP cloning experiment on MNIST
    python scripts/run_experiment.py \
      training=cloning \
      model=mlp \
      dataset=mnist \
      training.initial_epochs=20 \
      training.epochs_per_expansion=20 \
      training.expansion_factor=2 \
      training.num_expansions=1
    ;;
    
  cnn-cifar10)
    # CNN cloning experiment on CIFAR-10
    python scripts/run_experiment.py \
      training=cloning \
      model=cnn \
      dataset=cifar10 \
      training.initial_epochs=30 \
      training.epochs_per_expansion=30 \
      training.expansion_factor=2 \
      training.num_expansions=1
    ;;
    
  resnet-cifar10)
    # ResNet cloning experiment on CIFAR-10
    python scripts/run_experiment.py \
      training=cloning \
      model=resnet \
      dataset=cifar10 \
      training.initial_epochs=30 \
      training.epochs_per_expansion=30 \
      training.expansion_factor=2 \
      training.num_expansions=1
    ;;
    
  vit-cifar10)
    # Vision Transformer cloning experiment on CIFAR-10
    python scripts/run_experiment.py \
      training=cloning \
      model=vit \
      dataset=cifar10 \
      training.initial_epochs=30 \
      training.epochs_per_expansion=30 \
      training.expansion_factor=2 \
      training.num_expansions=1
    ;;
    
  multi-expansion)
    # Example with multiple expansion cycles
    python scripts/run_experiment.py \
      training=cloning \
      model=mlp \
      dataset=mnist \
      training.initial_epochs=15 \
      training.epochs_per_expansion=15 \
      training.expansion_factor=2 \
      training.num_expansions=3
    ;;
    
  custom)
    # Customize these parameters as needed
    MODEL="mlp"               # Options: mlp, cnn, resnet, vit
    DATASET="mnist"           # Options: mnist, cifar10, cifar100, tiny_imagenet
    INITIAL_EPOCHS=20
    EPOCHS_PER_EXPANSION=20
    EXPANSION_FACTOR=2
    NUM_EXPANSIONS=1
    USE_WANDB=false
    
    python scripts/run_experiment.py \
      training=cloning \
      model=$MODEL \
      dataset=$DATASET \
      training.initial_epochs=$INITIAL_EPOCHS \
      training.epochs_per_expansion=$EPOCHS_PER_EXPANSION \
      training.expansion_factor=$EXPANSION_FACTOR \
      training.num_expansions=$NUM_EXPANSIONS \
      logging.use_wandb=$USE_WANDB
    ;;
    
  *)
    echo "Unknown option: $OPTION"
    echo "Run './run_cloning_experiment.sh --help' for usage information"
    exit 1
    ;;
esac

echo "Experiment complete!"