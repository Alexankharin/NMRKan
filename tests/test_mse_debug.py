#!/usr/bin/env python3
"""Quick test to diagnose MSE N/A issue."""

import torch
import numpy as np
from run_many_experiments import ExperimentConfig, get_perturbation, make_dataset_from_function, train_kan_model
from nmrkan.models import KharKAN

config = ExperimentConfig()
print('Testing MSE value generation...')

# Create a small test dataset
data = make_dataset_from_function(100, get_perturbation, min_Jintra=-2, max_Jintra=-1, min_deltaJ=-0.7, max_deltaJ=-0.1, input_mode='3d')
print(f'Data shapes: input {data["train_input"].shape}, label {data["train_label"].shape}')

# Train a small model
model = KharKAN((3, 4, 3))
config.epochs = 100  # Just a few epochs for testing
loss_hist, mse_hist, rel_err_hist, final_mse = train_kan_model(model, data, config, 1e-4, 0.1, config.device)

print(f'Final MSE type: {type(final_mse)}')
print(f'Final MSE value: {final_mse}')
print(f'Final MSE repr: {repr(final_mse)}')

# Test the safe_format_float function
from report_utils import safe_format_float
formatted = safe_format_float(final_mse)
print(f'Formatted MSE: {formatted}')

# Test what happens during CSV creation
result_data = {
    'experiment_id': 'test',
    'architecture': (3, 4, 3),
    'lr': 1e-4,
    'l05_penalty': 0.1,
    'perturbation': {
        'final_mse': final_mse,
        'formulas': {'z_0': None, 'z_1': None, 'z_2': None},
        'refined_formulas': {'z_0': None, 'z_1': None, 'z_2': None},
        'symbolic_mse_scores': {'z_0': float('inf'), 'z_1': float('inf'), 'z_2': float('inf')},
        'total_complexity': 0
    }
}

from report_utils import extract_experiment_data_for_csv
csv_data = extract_experiment_data_for_csv([result_data])
print(f'CSV data sample:')
for key, value in csv_data[0].items():
    if 'mse' in key.lower():
        print(f'  {key}: {value} (type: {type(value)})')
