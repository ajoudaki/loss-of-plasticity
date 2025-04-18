# Neural Network Dynamic Scaling

This project investigates dynamic scaling properties of neural networks during training, with a particular focus on continual learning scenarios.

## Project Structure

```
project/
├── configs/                # Experiment configuration files
│   └── default_experiment.json
├── scripts/               # Utility scripts
│   ├── check_imports.py    # Check for import issues
│   ├── download_tiny_imagenet.py # Dataset download script
│   ├── extract_notebook.py # Extract content from Jupyter notebooks
│   └── run_experiment.py   # Script to run experiments
└── src/                   # Main source code
    ├── main.py             # Entry point
    ├── models/             # Model definitions
    │   ├── __init__.py
    │   ├── mlp.py          # MLP model
    │   ├── cnn.py          # CNN model
    │   ├── resnet.py       # ResNet model
    │   └── vit.py          # Vision Transformer model
    ├── utils/              # Utility modules
    │   ├── __init__.py
    │   ├── layers.py       # Layer utilities
    │   ├── metrics.py      # Metrics functions
    │   ├── monitor.py      # NetworkMonitor class
    │   ├── data.py         # Data loading utilities
    │   └── visualization.py # Plotting functions
    └── training/           # Training code
        ├── __init__.py
        ├── eval.py         # Evaluation functions
        └── train_continual.py # Continual learning training loop
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/NN-dynamic-scaling.git
   cd NN-dynamic-scaling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running an Experiment

Use the provided experiment script:

```bash
# Run with a configuration file
python scripts/run_experiment.py --config configs/default_experiment.json

# Or with command line arguments
python scripts/run_experiment.py --model cnn --dataset cifar10 --tasks 5 --epochs 20
```

### Downloading Datasets

For Tiny ImageNet:

```bash
python scripts/download_tiny_imagenet.py
```

### Running Directly

You can also run the main module directly:

```bash
python -m src.main --model cnn --dataset cifar10 --tasks 5 --epochs 20 --batch-size 128 --lr 0.001 --seed 42 --use-wandb
```

## Key Features

- Modular neural network architectures (MLP, CNN, ResNet, ViT)
- Comprehensive metrics for tracking network dynamics
- Continual learning experimentation framework
- Integration with Weights & Biases for experiment tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.