# Configuration System

This project uses Hydra for configuration management, which provides a hierarchical configuration system with composition, inheritance, and command-line overrides.

## Configuration Structure

The configuration is organized into the following directories:

- `model/`: Neural network model configurations
- `dataset/`: Dataset-specific configurations 
- `optimizer/`: Optimizer configurations
- `training/`: Training strategy configurations
- `metrics/`: Metrics collection and threshold configurations
- `experiment/`: Predefined experiment configurations

## Using the Configuration System

### Running with Default Configuration

```bash
python scripts/run_experiment.py
```

This will use the default configurations specified in `conf/config.yaml`.

### Overriding Configurations

You can override any configuration value via command line:

```bash
python scripts/run_experiment.py model=mlp dataset=mnist optimizer.lr=0.01
```

### Running Predefined Experiments

```bash
python scripts/run_experiment.py +experiment=mlp_mnist
```

### Creating Custom Tasks

You can define custom class partitions:

```bash
python scripts/run_experiment.py task.partitions=[[0,1],[2,3],[4,5,6],[7,8,9]]
```

### Multirun (Grid Search)

```bash
python scripts/run_experiment.py --multirun optimizer.lr=0.001,0.01,0.1 model=mlp,cnn
```

## Configuration Schema

All configuration parameters are defined with appropriate types in `src/config_schema.py`. This provides type checking and auto-completion in IDEs that support it.

## Adding New Configurations

To add a new configuration type:

1. Define a dataclass in `src/config_schema.py`
2. Create a new YAML file in the appropriate directory
3. Update the defaults in `conf/config.yaml` if necessary

## Configuration Inheritance

The system follows this hierarchy (from lowest to highest precedence):

1. Default configurations for each component
2. Dataset-specific configurations (affects model parameters)
3. Predefined experiment configurations (if specified) 
4. Command-line overrides