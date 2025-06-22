# NMR Symbolic Regression Repository
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Alexankharin/NMRKan)
This repository simulates NMR frequencies by solving eigenvalues of specific matrices and attempts to symbolically approximate the results, aiming to recover perturbation theory from simulated data.

## Overview
This project provides a framework for generating synthetic NMR (Nuclear Magnetic Resonance) frequency data, training neural-symbolic models to fit the data, and extracting interpretable symbolic formulas. The ultimate goal is to recover or approximate perturbation theory results from simulated data using symbolic regression.

## Structure
- `nmrkan/` - Core library with models and physics helpers
- `data/` - All datasets and raw data files
- `notebooks/` - Jupyter/VSCode notebooks for experiments and analysis
- `outputs/` - Figures and result outputs
- `run_symbolic_regressor.py` - Main entrypoint for running symbolic regression
- `run_grid_search.py` - Simple grid search experiment
- `genetic_KAN.py` - Evolutionary weight perturbation demo
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
  - scipy

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Main Notebook Workflow
The primary workflow is demonstrated in `notebooks/Single_run_end_to_end_test.ipynb`. This notebook walks through:
- Generating synthetic NMR datasets
- Training a Kolmogorov-Arnold Network (KAN) model
- Extracting symbolic expressions from the trained model
- Cleaning and interpreting the learned formulas
- Comparing neural and symbolic outputs
- Visualizing training progress and results

Each step is explained in detail, making it accessible for both programmers and non-programmers interested in symbolic regression and scientific machine learning.

## Why Symbolic Regression?
Symbolic regression provides interpretable, human-readable formulas that describe the relationships in your data. This is especially valuable in scientific domains, where understanding the underlying physics is as important as predictive accuracy. The approach used here aims to bridge the gap between black-box neural models and analytical theory.

## Usage
1. Install dependencies as above.
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
