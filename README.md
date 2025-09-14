# NMRKan - NMR Symbolic Regression Framework
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Alexankharin/NMRKan)

A unified framework for NMR frequency prediction using symbolic regression with Kolmogorov-Arnold Networks (KANs). This repository simulates NMR frequencies by solving eigenvalues of specific matrices and uses symbolic regression to recover interpretable formulas from the data.

## Overview
NMRKan provides a comprehensive toolkit for:
- Generating synthetic NMR frequency data from quantum mechanical simulations
- Training neural-symbolic models (KANs) to fit the data
- Extracting interpretable symbolic formulas from trained models
- Running systematic experiments with different data generation approaches
- Comparing regular and dimensionless formulations

The ultimate goal is to recover or approximate perturbation theory results from simulated data using symbolic regression, bridging the gap between black-box neural models and analytical theory.

## New Simplified Structure (2025)

After refactoring, the repository now has a cleaner, more organized structure with fewer entry points:

### 📁 Repository Organization
```
├── main.py                          # 🚀 MAIN ENTRY POINT - Unified CLI
├── experiments/
│   └── run_experiments.py          # Consolidated experiment runner
├── data_generation/                 # All data generation code
│   ├── __init__.py
│   ├── eigenvalues_4CH2.py         # 4CH2 system calculations  
│   ├── nmr_datagen.py              # General NMR data generation
│   └── datasets.py                 # Dataset creation utilities
├── nmrkan/                         # Core models and analysis
```

### 🎯 New Features

#### Selective Output Training (`output_numbers`)
The framework now supports training models on specific frequency outputs only:

```python
from experiments.run_experiments import UnifiedExperimentConfig

# Configure experiment to train on f2 and f0 frequencies only (skip f1)
config = UnifiedExperimentConfig()
config.output_numbers = (0, 2)  # Indices: 0=f2, 1=f1, 2=f0

# Neural network architecture automatically adjusts:
# Original: (2, 4, 3) -> Adjusted: (2, 4, 2)
# Training data is automatically filtered to selected outputs
```

**Benefits:**
- Focus on specific frequency relationships of interest
- Reduce model complexity for targeted analysis
- Compare performance across different output combinations
- Logs display selected outputs and architecture adjustments
│   ├── models.py                   # KAN model implementations
│   ├── dimensional_analysis.py     # Physics-aware analysis
│   └── sympy_torch.py             # SymPy integration
├── data/                           # Generated datasets
├── experiment_results/             # Experiment outputs
├── notebooks/                      # Analysis notebooks
├── tests/                          # Unit tests
└── legacy/                         # Old scripts (for reference)
    ├── run_many_experiments.py
    ├── run_many_experiments_dimensionless.py
    ├── run_symbolic_regressor.py
    ├── run_grid_search.py
    └── genetic_KAN.py
```

### 🎯 Simple Usage

The new unified interface makes everything accessible through a single command:

```bash
# Run all experiments (regular + dimensionless)
python main.py experiment

# Run only regular experiments  
python main.py experiment --mode regular

# Run only dimensionless experiments
python main.py experiment --mode dimensionless

# Run symbolic regression
python main.py symbolic

# Run grid search optimization
python main.py grid-search

# Run genetic algorithm optimization
python main.py genetic

# Show repository status
python main.py status

# List available configuration files
python main.py list-configs

# Get help
python main.py --help
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a quick experiment:**
   ```bash
   python main.py experiment --mode working
   ```

3. **Check results:**
   ```bash
   python main.py status
   ```

## Key Features

### 🔬 **Multiple Data Generation Approaches**
- **Regular**: Physical units with dimensional analysis
- **Dimensionless**: Scale-invariant formulations
- **4CH2 System**: Specialized 4-spin system calculations

### 🧠 **Advanced Neural Architecture**
- Kolmogorov-Arnold Networks (KANs) for symbolic regression
- Automatic symbolic formula extraction
- Physics-aware dimensional analysis

### 📊 **Comprehensive Experiments**
- Systematic parameter sweeps
- Multiple architecture comparisons  
- Automatic report generation
- CSV and JSON result exports

### 🎛️ **Flexible Configuration**
- Command-line interface for all operations
- Configurable experiment parameters
- Multiple optimization algorithms

## Migration from Old Structure

If you were using the old scripts directly:

| Old Command                                    | New Command                                      |
| ---------------------------------------------- | ------------------------------------------------ |
| `python run_many_experiments.py`               | `python main.py experiment --mode regular`       |
| `python run_many_experiments_dimensionless.py` | `python main.py experiment --mode dimensionless` |
| `python run_symbolic_regressor.py`             | `python main.py symbolic`                        |
| `python run_grid_search.py`                    | `python main.py grid-search`                     |
| `python genetic_KAN.py`                        | `python main.py genetic`                         |

## Requirements
- Python 3.9+
- PyTorch
- NumPy, SciPy
- SymPy
- Matplotlib
- See `requirements.txt` for complete list

## Scientific Background

### Why Symbolic Regression?
Symbolic regression provides interpretable, human-readable formulas that describe relationships in data. This is especially valuable in scientific domains where understanding the underlying physics is as important as predictive accuracy.

### NMR Frequency Prediction
The framework models NMR transition frequencies by:
1. Generating quantum mechanical Hamiltonians
2. Computing eigenvalues and transition frequencies
3. Training neural networks to learn the frequency patterns
4. Extracting symbolic expressions from the trained networks
5. Validating results against known perturbation theory

This approach helps bridge the gap between computational quantum mechanics and analytical theory, potentially discovering new approximations or confirming existing ones.

## Contributing

The repository now has a much cleaner structure. When contributing:
- Use the unified `main.py` interface for testing
- Add new data generation methods to `data_generation/`
- Core model improvements go in `nmrkan/`
- All experiments use the consolidated runner in `experiments/`

## Citation

If you use this code in your research, please cite our work on symbolic regression for quantum systems.
2. Run the symbolic regressor:
   ```bash
   python run_symbolic_regressor.py
   ```
   This generates synthetic data, trains the model, and extracts symbolic expressions. Results are saved as CSV and pickle files in the project directory.
3. Optionally try the grid search or GA demo:
   ```bash
   python run_grid_search.py
   python genetic_KAN.py
   ```
4. Explore and visualize results using the notebooks in the `notebooks/` directory, especially `Single_run_end_to_end_test.ipynb` for a full workflow example.

## Data
- All raw and processed data is stored in the `data/` directory. This includes synthetic datasets and results from previous runs.
- Notebooks and scripts expect data to be in this location.

## Outputs
- Figures and result files (e.g., plots, CSVs) are saved in the `outputs/` directory.

## Project Purpose
The project simulates NMR frequency data, then uses symbolic regression to extract analytical formulas, providing insight into perturbation theory from simulated results. The approach is general and can be adapted to other physical systems where symbolic regression is useful.

## Citation
If you use this code or ideas in your research, please cite appropriately. (Add citation information here if available.)

## Authors
- [Alexankhar]

## License


## Troubleshooting
- If you encounter missing package errors, ensure you have installed all dependencies from `requirements.txt`.
- For CUDA/GPU issues, set `--device cpu` when running the script if you do not have a compatible GPU.
- For questions or issues, please open an issue on the repository or contact the author.

## Contributing & Extending
Contributions are welcome! You can extend the project by:
- Adding new datasets or physical systems
- Improving model architectures or training routines
- Enhancing symbolic post-processing or interpretability
- Creating new analysis notebooks

Feel free to open issues or pull requests with suggestions or improvements.
