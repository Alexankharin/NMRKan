# NMR KAN Training Framework - Migration Guide

## Overview

The NMR KAN repository has been refactored into a unified training framework with the following improvements:

- **Single entry point**: `train.py` for all training operations
- **Modular architecture**: Separate trainers for single vs multi-output modes  
- **Flexible data generation**: Support for 4CH2 and NMR datagen methods
- **Automatic formula extraction**: Built-in symbolic regression analysis
- **Configuration management**: YAML-based configuration with CLI overrides

## New Structure

```
nmrkan/
├── training/          # Training framework
│   ├── base_trainer.py
│   ├── single_output_trainer.py
│   └── multi_output_trainer.py
├── data/              # Dataset management
│   ├── dataset_factory.py
│   └── loaders.py
├── config/            # Configuration classes
│   ├── training_config.py
│   ├── model_config.py
│   └── data_config.py
├── analysis/          # Formula extraction
│   └── formula_extractor.py
├── utils/             # Utilities
│   ├── metrics.py
│   ├── plotting.py
│   └── checkpoints.py
└── models.py          # Existing KAN models

configs/               # Configuration files
├── default.yaml
├── dimensionless.yaml
└── multi_output.yaml

train.py              # Unified entry point
```

## Migration from Old Scripts

### From `main.py` and `experiments/run_experiments.py`

**Old way:**
```bash
python main.py experiment --mode regular
python experiments/run_experiments.py --mode dimensionless
```

**New way:**
```bash
# Single model training
python train.py --dataset 4ch2 --mode single --outputs 3

# Dimensionless training  
python train.py --config configs/dimensionless.yaml

# Multi-output training
python train.py --config configs/multi_output.yaml
```

### From `run_many_experiments.py`

**Old way:**
```bash
python run_many_experiments.py
```

**New way:**
```bash
# Equivalent single training
python train.py --dataset 4ch2 --mode single --architecture 2,8,3,1

# Equivalent multi training
python train.py --dataset 4ch2 --mode multi --architecture 2,8,1 --outputs 3
```

### From `genetic_KAN.py`

**Old way:**
```python
# Manual model creation and training
model = KharKAN(layers=(2, 8, 3, 1))
# ... manual training loop
```

**New way:**
```python
# Use unified framework
from nmrkan import SingleOutputTrainer, TrainingConfig, ModelConfig, DataConfig

training_config = TrainingConfig(epochs=50000, learning_rate=1e-4)
model_config = ModelConfig(layers=(2, 8, 3, 1))
data_config = DataConfig(dataset_type='4ch2', num_samples=10000)

trainer = SingleOutputTrainer(training_config, model_config)
# ... use trainer.train()
```

## Key Features

### 1. Unified Training Interface

```bash
# Basic training
python train.py --dataset 4ch2 --mode single

# Advanced configuration
python train.py \
  --dataset nmr_datagen \
  --mode multi \
  --outputs 3 \
  --architecture 2,6,1 \
  --epochs 30000 \
  --learning-rate 5e-4 \
  --dimensionless
```

### 2. Configuration Files

Create custom configurations in YAML format:

```yaml
# my_config.yaml
training:
  epochs: 25000
  learning_rate: 2e-4
  training_mode: single

model:
  layers: [2, 10, 5, 3]

data:
  dataset_type: 4ch2
  num_samples: 15000
  dimensionless: true
```

Use with: `python train.py --config my_config.yaml`

### 3. Automatic Formula Extraction

The framework automatically extracts symbolic formulas from trained models:

- Generates LaTeX, Python, and text formats
- Calculates complexity metrics
- Compares with analytical formulas (if provided)
- Creates comprehensive analysis reports

### 4. Training Modes

**Single Output Mode:**
- One model predicts all outputs simultaneously
- Shared feature learning across outputs
- Faster training, potentially better generalization

**Multi Output Mode:**  
- Separate specialized models for each output
- Independent optimization per output
- Better handling of output-specific patterns

### 5. Dataset Options

**4CH2 Method:**
- Full 8×8 Hamiltonian eigenvalue computation
- High precision, slower generation
- Supports normalized eigenvalue outputs

**NMR Datagen Method:**
- Simplified 4×4 Hamiltonian approach
- Faster generation, good for quick experiments
- Based on perturbation theory validation

## Configuration Options

### Training Parameters

- `epochs`: Number of training epochs (default: 50000)
- `learning_rate`: Learning rate (default: 1e-4)
- `batch_size`: Batch size (default: 1000)
- `l05_penalty`: L0.5 regularization strength (default: 0.1)
- `optimizer`: Optimizer type ('adam' or 'sgd')
- `training_mode`: 'single' or 'multi'

### Model Architecture

- `layers`: Architecture tuple (e.g., [2, 8, 3, 1])
- `linear_bias`: Use bias in linear layers (default: true)
- `activations`: Custom activation functions (optional)

### Data Generation

- `dataset_type`: '4ch2' or 'nmr_datagen'
- `num_samples`: Number of samples to generate
- `dimensionless`: Use deltaJ/Jintra ratio as input
- `parameter_ranges`: Min/max values for Jintra and deltaJ
- `normalize_inputs`: Apply input normalization
- `num_outputs`: Output dimension (3 for frequencies, 8 for eigenvalues)

## Example Workflows

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create default config
python train.py --create-config

# Run basic training
python train.py --config configs/default.yaml
```

### Custom Experiment

```bash
# Multi-output with custom architecture
python train.py \
  --dataset 4ch2 \
  --mode multi \
  --architecture 2,12,6,1 \
  --outputs 3 \
  --epochs 40000 \
  --learning-rate 1e-4 \
  --l05-penalty 0.15 \
  --jintra-range 0.5,15.0 \
  --deltaj-range 0.005,2.0 \
  --output-dir my_experiment
```

### Dimensionless Training

```bash
# Train with deltaJ/Jintra ratio as input
python train.py \
  --config configs/dimensionless.yaml \
  --architecture 1,10,8,3 \
  --epochs 60000
```

## Output Files

After training, results are saved to `experiment_results/unified_experiments_<timestamp>/`:

- `best_model.pt`: Best model checkpoint
- `training_curves.png`: Training progress plots
- `formulas/`: Extracted symbolic formulas
  - `formula_extraction_report.md`: Comprehensive analysis
  - `formulas.latex`: LaTeX format formulas
  - `formulas.py`: Python code format
  - `formulas.txt`: Plain text format

## Backward Compatibility

The refactored framework maintains backward compatibility:

- Existing `nmrkan.models` classes unchanged
- Legacy scripts still functional
- Original `main.py` interface preserved
- Experiment results format compatible

## Performance Notes

- **Single mode**: Faster training, good for initial exploration
- **Multi mode**: Better specialization, potentially higher accuracy
- **4CH2 dataset**: More accurate but slower generation
- **NMR datagen**: Faster generation for quick iterations
- **Caching**: Datasets are cached for faster repeated experiments

## Troubleshooting

**Import errors**: Ensure the project root is in Python path
**CUDA errors**: Use `--device cpu` to force CPU training
**Memory issues**: Reduce `--batch-size` or `--num-samples`
**Slow training**: Try NMR datagen method or reduce model size

## Next Steps

1. Run example experiments with provided configurations
2. Customize configurations for your specific use case  
3. Analyze extracted formulas for physical insights
4. Compare single vs multi-output performance
5. Experiment with different architectures and regularization