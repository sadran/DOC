# AI Coding Agent Instructions for DOC

## Project Overview
**DOC** is a machine learning research framework investigating the Distribution of Classifiers (DOC) - analyzing how random neural network weights generalize on synthetic and real datasets. The project implements empirical risk minimization (ERM) experiments and classifier density estimation.

## Architecture & Key Components

### Entry Point & Execution Flow
- **`main.py`**: CLI entry point that loads YAML config and triggers experiment execution
  - Command: `python main.py --config configs/config.yaml`
  - Config format is strictly YAML with nested sections for experiment, model, dataset, training, plotting

### Core Module Dependencies
```
main.py
  ↓
ExperimentFactory (core/experiments.py)
  ├─ Creates GaussianClassificationExperiment (extensible for other types)
  ├─ Instantiates: MLP model, Gaussian dataset, Trainer, Evaluator, Plotter
  └─ run() orchestrates the full experiment pipeline
```

### Module Responsibilities

**`core/experiments.py`**: Orchestration layer
- `ExperimentFactory.create_experiment()` uses config['experiment']['type'] to instantiate experiments
- `GaussianClassificationExperiment.run()` coordinates the entire pipeline:
  1. Estimates classifier density D(E) by sampling random unit-sphere weights
  2. Trains models to zero error on varying dataset sizes
  3. Collects true error distribution and visualizes results

**`core/trainer.py`**: Training logic
- Implements ERM (empirical risk minimization) loops
- `train()`: Standard training loop
- `train_until_zero_error()`: Trains until training accuracy reaches 100%
- `random_sample_to_zero_error()`: Used in experiments for random weight sampling

**`core/dataset.py`**: Synthetic data generation
- `Gaussian`: Binary classification dataset with configurable feature dimension, sample count, and class separation
- Supports seeded generation for reproducibility
- Returns PyTorch Dataset interface: `(x: Tensor, y: Tensor)` pairs

**`models/base_network.py`**: Weight manipulation utilities
- `sample_unit_sphere_weights()`: Returns random weights normalized to unit sphere (critical for DOC estimation)
- `set_flatten_weights()`: Replaces all parameters with flattened weights (enables weight space exploration)
- `num_parameters()`: Total parameter count

**`models/mlp.py`**: Neural network architecture
- Fully-connected layers with configurable hidden layers
- Supports "relu" or "leaky_relu" activations
- Output layer includes activation (research-specific design choice)
- Extends `BaseNetwork` to inherit weight manipulation methods

**`core/evaluator.py`**: Evaluation utilities
- `compute_error()`: Computes misclassification rate on a DataLoader
- Returns float in [0, 1] range; uses `argmax` for predictions

**`utils/plotter.py`**: Visualization
- `plot_histogram()`: Density plots for error distributions
- `plot_boxplot()`: Box plots for comparing error distributions across conditions
- `show_plots()`: Displays all collected figures

**`utils/logger.py`**: Result tracking (under development)
- Intended to save experiment metadata and results

## Configuration Schema

Config file structure in `configs/config.yaml`:
```yaml
experiment:
  type: string           # "gaussian_classification" (extensible)
  name: string          # For logging
  model: string         # Key in models section
  dataset: string       # "gaussian", "mnist", or "caltech"
  seed: int            # For reproducibility

models:
  <model_name>:
    type: "mlp"
    input_dim: int
    hidden_layers: [int, ...]
    output_dim: int
    bias: bool
    activation: "relu" | "leaky_relu"

dataset:
  gaussian:
    n_classes: int
    feature_dim: int
    test_size: int
    mean_distance: float
    sigma: float

erm:
  n_values: [int, ...]     # Training set sizes to test
  solutions_per_n: int    # Samples per size

doc:
  n_trials: int           # Random weight samples for density estimation
  histogram_bins: int

training:
  optimizer: string
  lr: float
  batch_size: int
  epochs: int

plotting:
  save_plots: bool
  save_dir: string
```

## Patterns & Conventions

### Weight Space Exploration
This codebase treats weight space as a continuous domain:
- Weights are sampled uniformly on the unit sphere via `sample_unit_sphere_weights()`
- Weights are set via flattened vectors using `set_flatten_weights()`
- This pattern is core to estimating the Distribution of Classifiers

### Config-Driven Flexibility
- Experiments instantiate objects based on config values, not hardcoded types
- Model creation: `config['models'][config['experiment']['model']]`
- Allows testing multiple models/datasets without code changes

### PyTorch DataLoader Integration
- All datasets extend `torch.utils.data.Dataset`
- Training/evaluation code expects DataLoader objects with `(x, y)` tuple iteration
- Batch size is 32 by default in experiments (can be parameterized)

### Error as Float
- Error rates are returned as floats in [0, 1] range
- No class balancing applied - raw misclassification count / total samples

## Development Workflows

### Running Experiments
```bash
python main.py --config configs/config.yaml
```
- Modify config file for different model/dataset/hyperparameter combinations
- No other CLI flags; all configuration via YAML

### Extending with New Experiments
1. Create new experiment class inheriting `BaseExperiment` in `core/experiments.py`
2. Add case to `ExperimentFactory.create_experiment()` matching new config type
3. Add config section with experiment-specific parameters

### Adding New Models
1. Create model class inheriting `BaseNetwork` in `models/`
2. Implement `forward()` method
3. Reference in config: `models.<model_name>.type: "model_class_name"`

### Adding New Datasets
1. Create dataset class inheriting `torch.utils.data.Dataset` in `core/dataset.py`
2. Implement `__len__()` and `__getitem__()` returning `(Tensor, Tensor)` tuples
3. Add config section under `dataset.<dataset_name>`

## Critical Patterns to Preserve

- **Unit sphere weight sampling**: Always normalize weights before using in classifier density estimation
- **Config-first architecture**: Avoid hardcoding experiment parameters; use YAML config
- **Device agnostic**: Code supports CPU/GPU via device parameter in Evaluator
- **Reproducibility**: Seed parameters for data generation and model initialization
