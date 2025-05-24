# NMR Symbolic Regression Repository

This repository simulates NMR frequencies by solving eigenvalues of specific matrices and attempts to symbolically approximate the results, aiming to recover perturbation theory from simulated data.

## Structure
- `src/` - All source code modules (moved from `kharkan/`, `datagen/`, etc.)
- `data/` - All datasets and raw data files
- `notebooks/` - Jupyter/VSCode notebooks for experiments and analysis
- `outputs/` - Figures and result outputs
- `run_symbolic_regressor.py` - Main entrypoint for running symbolic regression

## Usage
1. Install dependencies (see requirements in code or notebooks).
2. Run the main script:
   ```
   python run_symbolic_regressor.py
   ```
3. Explore notebooks in the `notebooks/` directory for data analysis and experiments.

## Purpose
The project simulates NMR frequency data, then uses symbolic regression to extract analytical formulas, providing insight into perturbation theory from simulated results.

## Authors
- [Alexankhar]

## License
[Specify license here]
