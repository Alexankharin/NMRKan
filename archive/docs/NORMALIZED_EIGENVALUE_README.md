# Normalized Eigenvalue KAN Training Implementation

## Overview

This implementation adds support for training KAN models with normalized eigenvalue data, where:
- **Input**: 1 feature (deltaJ/Jintra ratio)
- **Output**: 8 features (normalized eigenvalues = eigenvalues/Jintra)

The implementation uses the properly ordered eigenvalue surfaces from `get_surfaces_vectors_in_region` to ensure consistency in eigenvalue ordering across parameter space.

## Key Components

### 1. `NormalizedEigenvalueSurface` Class
- **Location**: `data_generation/datasets.py`
- **Purpose**: Efficient computation and interpolation of normalized eigenvalue surfaces
- **Features**:
  - Computes high-resolution eigenvalue surfaces once using `get_surfaces_vectors_in_region`
  - Pre-computes normalized eigenvalues (eigenvalues/Jintra)
  - Provides fast interpolation for arbitrary parameter points
  - Configurable grid resolution for accuracy vs. speed trade-off

### 2. `make_normalized_eigenvalue_dataset` Function
- **Location**: `data_generation/datasets.py`
- **Purpose**: Generate training datasets for normalized eigenvalue KAN models
- **Features**:
  - Uses `NormalizedEigenvalueSurface` for efficient data generation
  - Supports configurable parameter ranges and grid resolution
  - Filters samples by ratio threshold to avoid extreme values
  - Returns PyTorch tensors ready for training

### 3. Main Command Integration
- **Command**: `python main.py normalized-eigenvalues`
- **Location**: `main.py`
- **Purpose**: Unified command-line interface for normalized eigenvalue experiments
- **Features**:
  - Calls experiment runner with normalized eigenvalue mode
  - Supports configuration files
  - Integrated with existing experiment infrastructure

### 4. Experiment Runner
- **Location**: `experiments/run_experiments.py`
- **Function**: `run_normalized_eigenvalue_experiments`
- **Features**:
  - Tests multiple KAN architectures (1 input, 8 outputs)
  - Supports different parameter sets (working, nonlinear, realistic, etc.)
  - Generates comprehensive reports and results
  - Uses the same experimental framework as other modes

## Usage Examples

### Basic Usage
```bash
# Train KAN models with normalized eigenvalue data
python main.py normalized-eigenvalues

# With custom configuration
python main.py normalized-eigenvalues --config my_config.json
```

### Programmatic Usage
```python
from data_generation import make_normalized_eigenvalue_dataset, NormalizedEigenvalueSurface

# Generate a dataset
dataset = make_normalized_eigenvalue_dataset(
    num_samples=1000,
    min_Jintra=-32,
    max_Jintra=-5,
    min_deltaJ=-15, 
    max_deltaJ=-0.1,
    grid_resolution=200
)

# Or use the surface class directly for custom applications
surface = NormalizedEigenvalueSurface(grid_resolution=300)
ratios, eigenvals = surface.get_ratio_and_eigenvalues(jintra_vals, deltaJ_vals)
```

## Performance Characteristics

- **Surface computation**: ~2-5 seconds for 100x100 to 200x200 grids
- **Interpolation**: Very fast (~0.0001 seconds for multiple points)
- **Total dataset generation**: ~5-10 seconds for 500-1000 samples with high-resolution surfaces
- **Memory usage**: Scales with grid resolution (200x200 grid uses ~50MB for all surfaces)

## Architecture Support

The implementation includes optimized KAN architectures for normalized eigenvalue prediction:
- (1, 8, 8): 1 input, 8 hidden neurons, 8 outputs
- (1, 12, 8): 1 input, 12 hidden neurons, 8 outputs  
- (1, 16, 8): 1 input, 16 hidden neurons, 8 outputs
- (1, 20, 8): 1 input, 20 hidden neurons, 8 outputs

## Key Advantages

1. **Proper Eigenvalue Ordering**: Uses `get_surfaces_vectors_in_region` to ensure consistent eigenvalue tracking across parameter space
2. **Efficiency**: Computes high-resolution surfaces once, then uses fast interpolation
3. **Flexibility**: Configurable grid resolution for accuracy vs. speed trade-off
4. **Integration**: Seamlessly integrated with existing experiment infrastructure
5. **Scalability**: Can handle large datasets with minimal computational overhead after initial surface computation

## Files Modified/Added

- `data_generation/datasets.py`: Added `NormalizedEigenvalueSurface` class and updated dataset functions
- `data_generation/__init__.py`: Updated exports
- `main.py`: Added normalized-eigenvalues command
- `experiments/run_experiments.py`: Added normalized eigenvalue experiment mode
- `test_normalized_eigenvalue.py`: Test script for validation

## Configuration

The implementation supports the same parameter sets as other experiment modes:
- `working`: Small parameter range for quick testing
- `nonlinear`: Broader range with nonlinear behavior  
- `realistic`: Physically realistic parameter ranges
- `bidirectional`: Full bidirectional parameter exploration
- `realistic_truncated`: Truncated realistic ranges