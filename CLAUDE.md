# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run experiment: `python scripts/run_experiment.py --config configs/default_experiment.json`
- Custom experiment: `python scripts/run_experiment.py --model [mlp/cnn/resnet/vit] --dataset [mnist/cifar10/cifar100/tiny-imagenet] --tasks 5`
- Dataset preparation: `python scripts/download_tiny_imagenet.py`
- Check imports: `python scripts/check_imports.py src`
- Validate models: Check model outputs with `torch.isnan()` to detect numerical issues

## Code Style Guidelines
- **Imports**: Standard library first, third-party second, local modules last
- **Indentation**: 4 spaces, ~80-100 character line length
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Google-style docstrings with parameters and returns sections
- **Type Annotations**: Use type hints in function signatures
- **Error Handling**: Use try-except with specific error messages, handle numerical edge cases

## Project Organization
- Neural network models in `src/models/` (MLP, CNN, ResNet, VisionTransformer)
- Training components in `src/training/` (train_continual.py, eval.py)
- Utility modules in `src/utils/` (data.py, layers.py, metrics.py, monitor.py)
- Configuration files in `configs/` directory (JSON format)
- Experiment tracking with Weights & Biases when available