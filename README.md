# Loss of plasticity 

This project investigates dynamic scaling properties of neural networks during training, with a particular focus on continual learning scenarios. It provides a framework for analyzing how neural networks adapt to new information and maintain previously learned knowledge.

## Project Structure

```
project/
├── conf/                  # Hydra configuration files
│   ├── config.yaml        # Main configuration
│   ├── model/             # Model configurations
│   │   ├── mlp.yaml
│   │   ├── cnn.yaml
│   │   ├── resnet.yaml
│   │   └── vit.yaml       # Vision Transformer config
│   ├── dataset/           # Dataset configurations
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   ├── cifar100.yaml
│   │   └── tiny_imagenet.yaml
│   ├── optimizer/         # Optimizer configurations
│   ├── metrics/           # Metrics configurations
│   └── training/          # Training configurations
├── data/                  # Directory for datasets
├── notebooks/             # Jupyter notebooks for analysis
│   ├── CL.ipynb           # Continual learning experiments
│   ├── coupling.ipynb     # Weight coupling analysis
│   └── main.ipynb         # Main experiments notebook
├── paper/                 # Academic paper materials
├── scripts/               # Utility scripts
│   ├── check_imports.py    # Check for import issues
│   ├── download_tiny_imagenet.py # Dataset download script
│   ├── extract_notebook.py # Extract content from Jupyter notebooks
│   └── run_experiment.py   # Main experiment script with Hydra
├── src/                   # Main source code
│   ├── config_schema.py    # Dataclass schemas for Hydra configs
│   ├── register_configs.py # Register configs with Hydra's ConfigStore
│   ├── continual_learning.py # CL-specific functionality
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   ├── mlp.py          # MLP model
│   │   ├── cnn.py          # CNN model
│   │   ├── resnet.py       # ResNet model
│   │   └── vit.py          # Vision Transformer model
│   ├── utils/              # Utility modules
│   │   ├── __init__.py
│   │   ├── layers.py       # Layer utilities
│   │   ├── metrics.py      # Metrics functions
│   │   ├── monitor.py      # NetworkMonitor class
│   │   ├── data.py         # Data loading utilities
│   │   └── visualization.py # Plotting functions
│   └── training/           # Training code
│       ├── __init__.py
│       ├── eval.py         # Evaluation functions
│       └── train_continual.py # Continual learning training loop
└── saved_models/          # Directory for saving trained models
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ajoudaki/loss-of-plasticity
   cd loss-of-plasticity
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Download the Tiny ImageNet dataset:
   ```bash
   python scripts/download_tiny_imagenet.py
   ```

## Usage

### Running an Experiment with Hydra

This project uses [Hydra](https://hydra.cc/) for configuration management, allowing for flexible and composable configuration overrides. Here are some examples of how to run experiments:

```bash
# Run with default configuration (CNN on CIFAR-10)
python scripts/run_experiment.py

# Change the model (use MLP instead of CNN)
python scripts/run_experiment.py model=mlp

# Change both model and dataset
python scripts/run_experiment.py model=mlp dataset=mnist

# Change specific parameters
python scripts/run_experiment.py optimizer=sgd optimizer.lr=0.01 training.batch_size=64

# Run for fewer epochs (dry run mode to test setup)
python scripts/run_experiment.py training.epochs_per_task=2 dryrun=true

# Enable Weights & Biases logging
python scripts/run_experiment.py logging.use_wandb=true

# Use Vision Transformer on CIFAR-100 with custom task setting
python scripts/run_experiment.py model=vit dataset=cifar100 task.tasks=10 task.classes_per_task=10
```

### Available Configurations

#### Models
- `model=mlp`: Multi-Layer Perceptron
- `model=cnn`: Convolutional Neural Network
- `model=resnet`: ResNet model
- `model=vit`: Vision Transformer

#### Datasets
- `dataset=mnist`: MNIST handwritten digits
- `dataset=cifar10`: CIFAR-10 image classification
- `dataset=cifar100`: CIFAR-100 image classification
- `dataset=tiny_imagenet`: Tiny ImageNet

#### Optimizers
- `optimizer=adam`: Adam optimizer (default)
- `optimizer=sgd`: Stochastic Gradient Descent
- `optimizer=rmsprop`: RMSProp optimizer

#### Training Settings
- `training.epochs_per_task`: Number of epochs per task
- `training.batch_size`: Batch size for training
- `training.seed`: Random seed for reproducibility

#### Task Settings
- `task.tasks`: Number of tasks in the continual learning sequence
- `task.classes_per_task`: Number of classes per task

#### Experiment Tracking
- `logging.use_wandb=true`: Enable Weights & Biases logging
- `logging.wandb_project="your-project"`: Set the W&B project name

## Key Features

- **Modular Architecture**: Easily swap between different neural network models
- **Dynamic Analysis**: Track metrics like weight norms, gradient magnitudes, and activation patterns
- **Continual Learning Framework**: Evaluate how networks adapt to sequences of tasks
- **Visualization Tools**: Plot learning trajectories, forgetting curves, and network dynamics
- **Configuration System**: Hydra-based configuration for reproducible experiments
- **Type Safety**: Structured configs with dataclasses providing validation and type checking

## Metrics

The framework tracks several metrics to analyze neural network dynamics during training. These metrics help identify various learning phenomena such as dead neurons, redundancy, and saturation.

### Dead Neurons (`dead_fraction`)
Measures the fraction of neurons that are inactive (producing zero or near-zero activations) across most input samples.
- **Definition**: A neuron is considered "dead" if it produces activations close to zero (abs < 1e-7) for more than the specified threshold (default: 95%) of input samples.
- **Configuration**: `metrics.dead_threshold` (default: 0.95)
- **Interpretation**: High values indicate neurons that aren't contributing to the network's function, suggesting wasted capacity or potential training issues.

### Duplicate Neurons (`dup_fraction`)
Measures the fraction of neurons that are functionally similar to other neurons in the same layer.
- **Definition**: A neuron is considered a "duplicate" if its normalized activation pattern has a correlation above the threshold (default: 0.95) with another neuron in the same layer.
- **Configuration**: `metrics.corr_threshold` (default: 0.95)
- **Interpretation**: High values suggest redundant representations and inefficient use of network capacity.

### Effective Rank (`eff_rank`)
Measures the effective dimensionality of neural representations as entropy of normalized singular values.
- **Definition**: The exponent of the entropy of the normalized singular values of the activation matrix.
- **Calculation**: `exp(-sum(p * log(p)))` where `p` are the normalized singular values.
- **Interpretation**: Indicates how many independent dimensions the network is effectively using for representing data. Higher values suggest more distributed and potentially more robust representations.

### Stable Rank (`stable_rank`)
A numerically stable approximation of the rank of activations.
- **Definition**: The ratio of squared Frobenius norm to squared spectral norm of the activation matrix.
- **Calculation**: `||A||_F^2 / ||A||_2^2` where `||A||_F` is the Frobenius norm and `||A||_2` is the spectral norm.
- **Interpretation**: Provides insight into how many significant singular values contribute to the representation. Higher values indicate more dimensions are being effectively utilized.

### Saturated Neurons (`saturated_frac`)
Measures the fraction of neurons that are saturated, meaning they have very small gradients relative to their activations.
- **Definition**: A neuron is considered "saturated" if the ratio of gradient magnitude to mean activation magnitude is below the saturation threshold for more than the specified percentage of samples.
- **Configuration**: 
  - `metrics.saturation_threshold` (default: 1e-4)
  - `metrics.saturation_percentage` (default: 0.99)
- **Interpretation**: High values indicate neurons whose weights are not being effectively updated during training, suggesting they may be stuck in flat regions of the loss landscape.

## Extending the Framework

### Adding a New Model

1. Create a new model class in `src/models/`
2. Add a configuration dataclass in `src/config_schema.py`
3. Register it in `src/register_configs.py`
4. Create a YAML config file in `conf/model/`

### Adding a New Dataset

1. Update the data loading utilities in `src/utils/data.py`
2. Add dataset configuration in `src/config_schema.py`
3. Create a YAML config file in `conf/dataset/`

## Notebooks

Explore the included Jupyter notebooks for analysis:

```bash
jupyter notebook notebooks/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
