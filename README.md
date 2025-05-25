# NMR Symbolic Regression Repository

This repository simulates NMR frequencies by solving eigenvalues of specific matrices and attempts to symbolically approximate the results, aiming to recover perturbation theory from simulated data.

## Overview
This project provides a framework for generating synthetic NMR (Nuclear Magnetic Resonance) frequency data, training neural-symbolic models to fit the data, and extracting interpretable symbolic formulas. The ultimate goal is to recover or approximate perturbation theory results from simulated data using symbolic regression.

## Structure
- `src/` - All source code modules (moved from `kharkan/`, `datagen/`, etc.)
- `data/` - All datasets and raw data files
- `notebooks/` - Jupyter/VSCode notebooks for experiments and analysis
- `outputs/` - Figures and result outputs
- `run_symbolic_regressor.py` - Main entrypoint for running symbolic regression
- `requirements.txt` - List of required Python packages

## Requirements
- Python 3.9+
- See `requirements.txt` for all dependencies:
  - numpy
  - pandas
  - torch
  - sympy
  - optuna
  - tqdm
  - matplotlib
  - seaborn

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
1. Install dependencies as above.
2. Run the main script:
   ```bash
   python run_symbolic_regressor.py
   ```
   This will generate synthetic data, train the model, and perform symbolic regression. Results are saved as CSV and pickle files in the project directory.
3. Explore and visualize results using the notebooks in the `notebooks/` directory.

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
