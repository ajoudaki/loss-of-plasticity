# Neural Network Dynamic Scaling Scripts

This directory contains utility scripts for the Neural Network Dynamic Scaling project.

## Available Scripts

### `run_experiment.py`

A convenient script for running experiments with different configurations.

```bash
# Run with a JSON configuration file
python scripts/run_experiment.py --config configs/my_experiment.json

# Run with command line arguments
python scripts/run_experiment.py --model cnn --dataset cifar10 --tasks 5 --epochs 20 --batch-size 128 --lr 0.001 --seed 42 --use-wandb
```

### `download_tiny_imagenet.py`

Downloads and extracts the Tiny ImageNet dataset.

```bash
python scripts/download_tiny_imagenet.py
```

### `extract_notebook.py`

Extracts content from Jupyter notebooks for easier review.

```bash
# Extract from all notebooks in a directory
python scripts/extract_notebook.py path/to/notebooks

# Extract with specific file extensions
python scripts/extract_notebook.py path/to/directory --extensions .ipynb,.py
```

### `check_imports.py`

Scans Python files for potential import issues.

```bash
# Check imports in the src directory
python scripts/check_imports.py src

# Include system modules in the check
python scripts/check_imports.py src --check-system-modules
```

## Creating a Configuration File

Configuration files should be JSON files with parameter key-value pairs. Example:

```json
{
    "model": "cnn",
    "dataset": "cifar10",
    "tasks": 5,
    "epochs": 20,
    "batch_size": 128,
    "lr": 0.001,
    "seed": 42,
    "use_wandb": true
}
```

Save these files in a `configs` directory for organizational purposes.