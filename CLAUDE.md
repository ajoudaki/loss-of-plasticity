# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Main script: `python scripts/run_experiment.py`
- Custom experiment: `python scripts/run_experiment.py model=mlp dataset=mnist`
- Dataset preparation: `python scripts/download_tiny_imagenet.py`
- Extract notebook: `python scripts/extract_notebook.py path/to/notebooks`
- Check imports: `python scripts/check_imports.py`
- Lint: `flake8 src/ scripts/`
- Type check: `mypy --ignore-missing-imports src/`

## Code Style Guidelines
- **Imports**: Standard library first, third-party second, local modules last
- **Indentation**: 4 spaces, ~80-100 character line length
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Documentation**: Google-style docstrings with parameters, returns, and descriptions
- **Type Annotations**: Use type hints for function parameters and return values
- **Error Handling**: Use try-except with specific error messages, handle numerical edge cases
- **Validation**: Check model outputs with `torch.isnan()` to detect numerical issues

## Project Organization
- Models in `src/models/` (MLP, CNN, ResNet, ViT implementations)
- Training components in `src/training/` (train routines, evaluation)
- Utility modules in `src/utils/` (data loading, metrics, visualization)
- Configuration files in `conf/` directory (YAML format with Hydra)
- Dependencies: PyTorch, NumPy, Matplotlib, Hydra, OmegaConf, Wandb (optional)