#!/usr/bin/env python3
"""
Unified KAN experiment runner for NMR spectroscopy data.

This script runs multiple KAN experiments with different architectures and parameters,
supporting both regular and dimensionless approaches. It generates comprehensive
reports with symbolic expressions, training curves, and surface plots saved as PDF and CSV.

Usage:
    python experiments/run_experiments.py [--mode regular|dimensionless|both] [--config CONFIG]
"""

import sys
import time
import json
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sympy import Expr

# Add the current directory to the path so we can import our custom modules
if "." not in sys.path:
    sys.path.append(".")

from nmrkan.models import KharKAN, _clean_expr, _fix_powers, _round_expr
from run_many_experiments import apply_dimensional_fixing, calculate_formula_complexity

from data_generation import (
    make_dataset_from_function,
    make_dimensionless_dataset_from_function,
    get_eigenvalue_dimensionless,
    get_perturbation_dimensionless,
    get_perturbation,
    get_frequences_ordered,
    make_normalized_eigenvalue_dataset,
)
from report_utils import save_results_to_csv, print_console_summary, print_detailed_console_summary

warnings.filterwarnings('ignore', category=UserWarning)
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')


class UnifiedExperimentConfig:
    """Configuration class for unified experiments."""
    
    def __init__(self):
        # Model architectures to test
        self.regular_architectures = [
            (2, 4, 3),  # 2 inputs (deltaJ, deltaJ/Jintra), 4 hidden, 3 outputs
            (2, 6, 3),  # 2 inputs, 6 hidden, 3 outputs
            (2, 8, 3),  # 2 inputs, 8 hidden, 3 outputs
            (3, 4, 3),  # 3 inputs (deltaJ, deltaJ/Jintra, Jintra), 4 hidden, 3 outputs
            (3, 6, 3),  # 3 inputs, 6 hidden, 3 outputs
            (3, 8, 3),  # 3 inputs, 8 hidden, 3 outputs
            (4, 4, 3),  # 4 inputs (deltaJ, deltaJ/Jintra, Jintra, Jintra/deltaJ), 4 hidden, 3 outputs
            (4, 6, 3),  # 4 inputs, 6 hidden, 3 outputs
            (4, 8, 3),  # 4 inputs, 8 hidden, 3 outputs
        ]
        
        self.dimensionless_architectures = [
            (1, 4, 3),  # 1 input (dJ/J), 4 hidden, 3 outputs
            (1, 6, 3),  # 1 input, 6 hidden, 3 outputs
            (1, 8, 3),  # 1 input, 8 hidden, 3 outputs
            (2, 4, 3),  # 2 inputs (dJ/J, J/dJ), 4 hidden, 3 outputs
            (2, 6, 3),  # 2 inputs, 6 hidden, 3 outputs
            (2, 8, 3),  # 2 inputs, 8 hidden, 3 outputs
        ]
        
        self.normalized_eigenvalue_architectures = [
            (1, 8, 8),   # 1 input (dJ/J), 8 hidden, 8 outputs (8 normalized eigenvalues)
            (1, 12, 8),  # 1 input, 12 hidden, 8 outputs
            (1, 16, 8),  # 1 input, 16 hidden, 8 outputs  
            (1, 20, 8),  # 1 input, 20 hidden, 8 outputs
        ]
        
        # Training parameters to test
        self.learning_rates = [1e-4]
        self.l05_penalties = [0.1]
        self.epochs = 50000
        self.early_stop_threshold = 1e-4
        
        # Data generation parameters
        self.data_params = {
            "working": {
                "MIN_JINTRA": -2,
                "MAX_JINTRA": -1,
                "MIN_DELTAJ": -0.7,
                "MAX_DELTAJ": -0.1,
                "RATIO_THRESHOLD": 100,
            },
            "nonlinear": {
                "MIN_JINTRA": -32,
                "MAX_JINTRA": -5,
                "MIN_DELTAJ": -15,
                "MAX_DELTAJ": -0.1,
                "RATIO_THRESHOLD": 100,
            },
            "realistic": {
                "MIN_JINTRA": -15,
                "MAX_JINTRA": -12,
                "MIN_DELTAJ": -7,
                "MAX_DELTAJ": -0.2,
                "RATIO_THRESHOLD": 100,
            },
            "bidirectional": {
                "MIN_JINTRA": -10,
                "MAX_JINTRA": 10,
                "MIN_DELTAJ": -1,
                "MAX_DELTAJ": 1,
                "RATIO_THRESHOLD": 100,
            },
            "realistic_truncated": {
                "MIN_JINTRA": -15,
                "MAX_JINTRA": -12,
                "MIN_DELTAJ": -2,
                "MAX_DELTAJ": -0.2,
                "RATIO_THRESHOLD": 100,
            },
        }
        
        self.num_samples = 20000
        self.batch_size = None  # Use full batch (no batching)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_symbolic_formulas(model: KharKAN, num_inputs: int) -> Dict[str, Optional[Expr]]:
    """Extract symbolic formulas from a trained KAN model."""
    formula_dict = {}
    
    try:
        for output_idx in range(3):  # 3 outputs: z_0, z_1, z_2
            symbolic_formula = model.symbolic_formula(var=[f'x_{i}' for i in range(num_inputs)])[0][output_idx]
            symbolic_formula = _clean_expr(_fix_powers(_round_expr(symbolic_formula)))
            formula_dict[f'z_{output_idx}'] = symbolic_formula
    except Exception as e:
        print(f"Warning: Could not extract symbolic formula: {e}")
        for output_idx in range(3):
            formula_dict[f'z_{output_idx}'] = None
    
    return formula_dict


def train_kan_model(
    model: KharKAN,
    data: Dict[str, torch.Tensor],
    config: UnifiedExperimentConfig,
    lr: float,
    l05_penalty: float,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], float]:
    """Train a KAN model and return training history."""
    
    model.to(device)
    
    # Prepare data
    inputs = data['train_input'].to(device)
    targets = data['train_label'].to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    loss_history = []
    mse_history = []
    rel_err_history = []
    
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate losses
        mse_loss = nn.MSELoss()(outputs, targets)
        l05_reg = sum(torch.norm(p, 0.5) for p in model.parameters())
        total_loss = mse_loss + l05_penalty * l05_reg
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        loss_history.append(total_loss.item())
        mse_history.append(mse_loss.item())
        
        with torch.no_grad():
            rel_err = torch.mean(torch.abs((outputs - targets) / (targets + 1e-8)))
            rel_err_history.append(rel_err.item())
        
        # Early stopping
        if mse_loss.item() < config.early_stop_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Print progress occasionally
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}: MSE={mse_loss.item():.2e}, Total Loss={total_loss.item():.2e}")
    
    return loss_history, mse_history, rel_err_history, mse_history[-1]


def run_single_experiment(
    architecture: Tuple[int, ...],
    data_pert: Dict[str, torch.Tensor],
    data_eigen: Dict[str, torch.Tensor],
    config: UnifiedExperimentConfig,
    lr: float,
    l05_penalty: float,
    experiment_id: str,
    params_name: str = "",
    is_dimensionless: bool = False
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    results = {
        'experiment_id': experiment_id,
        'architecture': architecture,
        'lr': lr,
        'l05_penalty': l05_penalty,
        'timestamp': time.time(),
        'is_dimensionless': is_dimensionless
    }
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_id}")
    print(f"Architecture: {architecture}, LR: {lr}, L05: {l05_penalty}")
    print(f"Using parameter set: {params_name}")
    print(f"Mode: {'Dimensionless' if is_dimensionless else 'Regular'}")
    print(f"{'='*60}")
    
    # Run perturbation theory experiment (skip if data_pert is None)
    if data_pert is not None:
        print("\n--- Perturbation Theory Data ---")
        model_pert = KharKAN(architecture)
        
        loss_hist_pert, mse_hist_pert, rel_err_hist_pert, final_mse_pert = train_kan_model(
            model_pert, data_pert, config, lr, l05_penalty, config.device
        )
        
        formulas_pert = extract_symbolic_formulas(model_pert, architecture[0])
        complexity_pert = calculate_formula_complexity(formulas_pert)
        
        # Apply dimensional analysis to fix formulas and optimize coefficients (only for regular mode)
        if not is_dimensionless:
            print("Applying dimensional analysis to fix formulas...")
            (
                refined_formulas_pert,
                consistency_flags_pert,
                fix_messages_pert,
                optimization_results_pert,
            ) = apply_dimensional_fixing(
                formulas_pert,
                architecture[0],
                x_data=data_pert["train_input"],
                y_data={
                    "z_0": data_pert["train_label"][:, 0],
                    "z_1": data_pert["train_label"][:, 1],
                    "z_2": data_pert["train_label"][:, 2],
                },
                expected_output_dim="Hz",
            )
        else:
            # For dimensionless experiments, no dimensional analysis needed
            refined_formulas_pert = formulas_pert
            consistency_flags_pert = {name: True for name in formulas_pert.keys()}
            fix_messages_pert = {name: "Dimensionless - no fixing needed" for name in formulas_pert.keys()}
            optimization_results_pert = {name: {"success": True, "message": "Dimensionless"} for name in formulas_pert.keys()}
        
        results["perturbation"] = {
            "final_mse": final_mse_pert,
            "loss_history": loss_hist_pert,
            "mse_history": mse_hist_pert,
            "rel_err_history": rel_err_hist_pert,
            "formulas": formulas_pert,
            "refined_formulas": refined_formulas_pert,
            "dimensional_consistency": consistency_flags_pert,
            "dimensional_messages": fix_messages_pert,
            "optimization_results": optimization_results_pert,
            "complexity": complexity_pert,
            "total_complexity": sum(
                c for c in complexity_pert.values() if c != float("inf")
            ),
        }
        
        print(f"Perturbation - Final MSE: {final_mse_pert:.2e}")
        for name, formula in formulas_pert.items():
            if formula is not None:
                print(f"{name}: {formula}")
    else:
        print("\n--- Skipping Perturbation Theory (not applicable for normalized eigenvalues) ---")
        results["perturbation"] = None
    
    # Run eigenvalue experiment
    print("\n--- Eigenvalue Data ---")
    model_eigen = KharKAN(architecture)
    
    loss_hist_eigen, mse_hist_eigen, rel_err_hist_eigen, final_mse_eigen = (
        train_kan_model(model_eigen, data_eigen, config, lr, l05_penalty, config.device)
    )
    
    formulas_eigen = extract_symbolic_formulas(model_eigen, architecture[0])
    complexity_eigen = calculate_formula_complexity(formulas_eigen)
    
    # Apply dimensional analysis to fix formulas and optimize coefficients (only for regular mode)
    if not is_dimensionless:
        print("Applying dimensional analysis to fix formulas...")
        
        # Create y_data dictionary based on the number of outputs
        num_outputs = data_eigen["train_label"].shape[1]
        y_data = {}
        for i in range(num_outputs):
            y_data[f"z_{i}"] = data_eigen["train_label"][:, i]
        
        (
            refined_formulas_eigen,
            consistency_flags_eigen,
            fix_messages_eigen,
            optimization_results_eigen,
        ) = apply_dimensional_fixing(
            formulas_eigen,
            architecture[0],
            x_data=data_eigen["train_input"],
            y_data=y_data,
            expected_output_dim="Hz",
        )
    else:
        # For dimensionless experiments, no dimensional analysis needed
        refined_formulas_eigen = formulas_eigen
        consistency_flags_eigen = {name: True for name in formulas_eigen.keys()}
        fix_messages_eigen = {name: "Dimensionless - no fixing needed" for name in formulas_eigen.keys()}
        optimization_results_eigen = {name: {"success": True, "message": "Dimensionless"} for name in formulas_eigen.keys()}
    
    results["eigenvalue"] = {
        "final_mse": final_mse_eigen,
        "loss_history": loss_hist_eigen,
        "mse_history": mse_hist_eigen,
        "rel_err_history": rel_err_hist_eigen,
        "formulas": formulas_eigen,
        "refined_formulas": refined_formulas_eigen,
        "dimensional_consistency": consistency_flags_eigen,
        "dimensional_messages": fix_messages_eigen,
        "optimization_results": optimization_results_eigen,
        "complexity": complexity_eigen,
        "total_complexity": sum(
            c for c in complexity_eigen.values() if c != float("inf")
        ),
    }
    
    print(f"Eigenvalue - Final MSE: {final_mse_eigen:.2e}")
    for name, formula in formulas_eigen.items():
        if formula is not None:
            print(f"{name}: {formula}")
    
    return results


def save_results_to_json(results: List[Dict], output_dir: Path):
    """Save experiment results to JSON file."""
    json_path = output_dir / "results.json"
    
    # Convert numpy arrays and torch tensors to lists for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                serializable_value = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        serializable_value[k] = v.tolist() if hasattr(v, 'tolist') else list(v)
                    elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (float, int)):
                        serializable_value[k] = list(v)
                    else:
                        serializable_value[k] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                serializable_result[key] = serializable_value
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                serializable_result[key] = value.tolist() if hasattr(value, 'tolist') else list(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {json_path}")


def run_regular_experiments(config: UnifiedExperimentConfig, output_dir: Path, param_names: List[str]) -> List[Dict]:
    """Run regular (non-dimensionless) experiments."""
    print("\n" + "="*80)
    print("RUNNING REGULAR EXPERIMENTS")
    print("="*80)
    
    all_datasets = {}
    all_results = []
    
    # Generate datasets for all parameter sets
    print("\nGenerating regular datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Create datasets with different input dimensions for this parameter set
        datasets_2d = {}
        datasets_3d = {}
        datasets_4d = {}
        
        for arch in config.regular_architectures:
            input_dim = arch[0]

            if input_dim == 2 and "2d" not in datasets_2d:
                # Generate 2-input datasets
                data_eigen_2d = make_dataset_from_function(
                    config.num_samples,
                    get_frequences_ordered,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="2d",
                )

                # Generate perturbation data with same inputs
                deltaJ_vals = data_eigen_2d["train_input"][:, 0:1].numpy()
                ratio_vals = data_eigen_2d["train_input"][:, 1:2].numpy()
                jintra_vals = deltaJ_vals / ratio_vals

                data_pert_2d = {
                    "train_input": data_eigen_2d["train_input"].clone(),
                    "train_label": torch.tensor(
                        get_perturbation(jintra_vals, deltaJ_vals)
                    ).float(),
                }

                datasets_2d = {"pert": data_pert_2d, "eigen": data_eigen_2d}
                
            elif input_dim == 3 and "3d" not in datasets_3d:
                # Generate 3-input datasets
                data_eigen_3d = make_dataset_from_function(
                    config.num_samples,
                    get_frequences_ordered,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="3d",
                )
                
                data_pert_3d = {
                    "train_input": data_eigen_3d["train_input"].clone(),
                    "train_label": torch.tensor(
                        get_perturbation(
                            data_eigen_3d["train_input"][:, 2:3].numpy(),  # x2 = Jintra
                            data_eigen_3d["train_input"][:, 0:1].numpy(),  # x0 = deltaJ
                        )
                    ).float(),
                }
                
                datasets_3d = {'pert': data_pert_3d, 'eigen': data_eigen_3d}
                
            elif input_dim == 4 and '4d' not in datasets_4d:
                # Generate 4-input datasets
                data_eigen_4d = make_dataset_from_function(
                    config.num_samples,
                    get_frequences_ordered,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="4d",
                )
                
                data_pert_4d = {
                    "train_input": data_eigen_4d["train_input"].clone(),
                    "train_label": torch.tensor(
                        get_perturbation(
                            data_eigen_4d["train_input"][:, 2:3].numpy(),  # x2 = Jintra
                            data_eigen_4d["train_input"][:, 0:1].numpy(),  # x0 = deltaJ
                        )
                    ).float(),
                }
                
                datasets_4d = {'pert': data_pert_4d, 'eigen': data_eigen_4d}
        
        # Store datasets for this parameter set
        all_datasets[params_name] = {
            "2d": datasets_2d,
            "3d": datasets_3d,
            "4d": datasets_4d,
        }
    
    # Run experiments
    print("\nStarting regular experiments...")
    experiment_count = 0
    total_experiments = len(param_names) * len(config.regular_architectures) * len(config.learning_rates) * len(config.l05_penalties)
    
    for params_name in param_names:
        datasets_2d = all_datasets[params_name]["2d"]
        datasets_3d = all_datasets[params_name]['3d']
        datasets_4d = all_datasets[params_name]['4d']
        
        for architecture in config.regular_architectures:
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    experiment_count += 1
                    experiment_id = f"regular_{params_name}_arch_{architecture}_lr_{lr}_l05_{l05_penalty}"
                    
                    print(f"\nExperiment {experiment_count}/{total_experiments}: {experiment_id}")
                    
                    # Select appropriate dataset based on input dimension
                    if architecture[0] == 2:
                        data_pert = datasets_2d["pert"]
                        data_eigen = datasets_2d["eigen"]
                    elif architecture[0] == 3:
                        data_pert = datasets_3d['pert']
                        data_eigen = datasets_3d['eigen']
                    else:  # 4 inputs
                        data_pert = datasets_4d['pert']
                        data_eigen = datasets_4d['eigen']
                    
                    try:
                        result = run_single_experiment(
                            architecture, data_pert, data_eigen, config, lr, l05_penalty, 
                            experiment_id, params_name, is_dimensionless=False
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in experiment {experiment_id}: {e}")
                        all_results.append({
                            'experiment_id': experiment_id,
                            'architecture': architecture,
                            'lr': lr,
                            'l05_penalty': l05_penalty,
                            'params_name': params_name,
                            'is_dimensionless': False,
                            'error': str(e)
                        })
    
    return all_results


def run_dimensionless_experiments(config: UnifiedExperimentConfig, output_dir: Path, param_names: List[str]) -> List[Dict]:
    """Run dimensionless experiments."""
    print("\n" + "="*80)
    print("RUNNING DIMENSIONLESS EXPERIMENTS")
    print("="*80)
    
    all_datasets = {}
    all_results = []
    
    # Generate datasets for all parameter sets
    print("\nGenerating dimensionless datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating dimensionless data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Create datasets with different input dimensions for this parameter set
        datasets_1d = {}
        datasets_2d = {}
        
        for arch in config.dimensionless_architectures:
            input_dim = arch[0]

            if input_dim == 1 and "1d" not in datasets_1d:
                # Generate 1-input datasets
                data_eigen_1d = make_dimensionless_dataset_from_function(
                    config.num_samples,
                    get_eigenvalue_dimensionless,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="1d",
                )

                data_pert_1d = make_dimensionless_dataset_from_function(
                    config.num_samples,
                    get_perturbation_dimensionless,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="1d",
                )

                datasets_1d = {"pert": data_pert_1d, "eigen": data_eigen_1d}

            elif input_dim == 2 and "2d" not in datasets_2d:
                # Generate 2-input datasets
                data_eigen_2d = make_dimensionless_dataset_from_function(
                    config.num_samples,
                    get_eigenvalue_dimensionless,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="2d",
                )

                data_pert_2d = make_dimensionless_dataset_from_function(
                    config.num_samples,
                    get_perturbation_dimensionless,
                    min_Jintra=params["MIN_JINTRA"],
                    max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"],
                    max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"],
                    input_mode="2d",
                )
                
                datasets_2d = {'pert': data_pert_2d, 'eigen': data_eigen_2d}
        
        # Store datasets for this parameter set
        all_datasets[params_name] = {
            "1d": datasets_1d,
            "2d": datasets_2d,
        }
    
    # Run experiments
    print("\nStarting dimensionless experiments...")
    experiment_count = 0
    total_experiments = len(param_names) * len(config.dimensionless_architectures) * len(config.learning_rates) * len(config.l05_penalties)
    
    for params_name in param_names:
        datasets_1d = all_datasets[params_name]["1d"]
        datasets_2d = all_datasets[params_name]['2d']
        
        for architecture in config.dimensionless_architectures:
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    experiment_count += 1
                    experiment_id = f"dimensionless_{params_name}_arch_{architecture}_lr_{lr}_l05_{l05_penalty}"
                    
                    print(f"\nExperiment {experiment_count}/{total_experiments}: {experiment_id}")
                    
                    # Select appropriate dataset based on input dimension
                    if architecture[0] == 1:
                        data_pert = datasets_1d["pert"]
                        data_eigen = datasets_1d["eigen"]
                    elif architecture[0] == 2:
                        data_pert = datasets_2d['pert']
                        data_eigen = datasets_2d['eigen']
                    else:
                        print(f"Skipping unsupported architecture: {architecture}")
                        continue
                    
                    try:
                        result = run_single_experiment(
                            architecture, data_pert, data_eigen, config, lr, l05_penalty, 
                            experiment_id, params_name, is_dimensionless=True
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in experiment {experiment_id}: {e}")
                        all_results.append({
                            'experiment_id': experiment_id,
                            'architecture': architecture,
                            'lr': lr,
                            'l05_penalty': l05_penalty,
                            'params_name': params_name,
                            'is_dimensionless': True,
                            'error': str(e)
                        })
    
    return all_results


def run_normalized_eigenvalue_experiments(config: UnifiedExperimentConfig, output_dir: Path, param_names: List[str]) -> List[Dict]:
    """Run normalized eigenvalue experiments with 1 input (deltaJ/Jintra) and 8 outputs (normalized eigenvalues)."""
    print("\n" + "="*80)
    print("RUNNING NORMALIZED EIGENVALUE EXPERIMENTS")
    print("="*80)
    
    all_results = []
    
    # Generate datasets for all parameter sets
    print("\nGenerating normalized eigenvalue datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Generate dataset with 1 input (deltaJ/Jintra) and 8 outputs (normalized eigenvalues)
        normalized_eigen_data = make_normalized_eigenvalue_dataset(
            config.num_samples,
            min_Jintra=params["MIN_JINTRA"],
            max_Jintra=params["MAX_JINTRA"],
            min_deltaJ=params["MIN_DELTAJ"],
            max_deltaJ=params["MAX_DELTAJ"],
            ratio_threshold=params["RATIO_THRESHOLD"],
        )
        
        # Test all normalized eigenvalue architectures
        for arch in config.normalized_eigenvalue_architectures:
            input_dim, hidden_dim, output_dim = arch
            
            # Verify input and output dimensions match our dataset
            assert input_dim == normalized_eigen_data["train_input"].shape[1], f"Input dimension mismatch: {input_dim} vs {normalized_eigen_data['train_input'].shape[1]}"
            assert output_dim == normalized_eigen_data["train_label"].shape[1], f"Output dimension mismatch: {output_dim} vs {normalized_eigen_data['train_label'].shape[1]}"
            
            # Test different learning rates and L0.5 penalties
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    
                    print(f"\nTesting architecture {arch} with lr={lr}, l05_penalty={l05_penalty}")
                    print(f"Dataset: {params_name}, Target: normalized_eigenvalues")
                    print(f"Input shape: {normalized_eigen_data['train_input'].shape}")
                    print(f"Output shape: {normalized_eigen_data['train_label'].shape}")
                    
                    result = run_single_experiment(
                        architecture=arch,
                        data_pert=None,  # Not used for normalized eigenvalues
                        data_eigen=normalized_eigen_data,  # Use our normalized eigenvalue data
                        config=config,
                        lr=lr,
                        l05_penalty=l05_penalty,
                        experiment_id=f"normalized_eigen_{params_name}_{arch}_{lr}_{l05_penalty}",
                        params_name=params_name,
                        is_dimensionless=True  # These are dimensionless normalized values
                    )
                    
                    all_results.append(result)
    
    return all_results


def main():
    """Main function to run unified experiments."""
    parser = argparse.ArgumentParser(description="Run unified KAN experiments")
    parser.add_argument("--mode", choices=["regular", "dimensionless", "both", "normalized-eigenvalues"], 
                       default="both", help="Experiment mode to run")
    parser.add_argument("--config", type=str, help="Path to config file (future feature)")
    
    args = parser.parse_args()
    
    # Setup
    config = UnifiedExperimentConfig()
    output_dir = Path('experiment_results') / f"unified_experiments_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {config.device}")
    print(f"Output directory: {output_dir}")
    print(f"Experiment mode: {args.mode}")
    
    # Parameter sets to test
    param_names = [
        "working",
        "nonlinear", 
        "realistic",
        "bidirectional",
        "realistic_truncated",
    ]
    
    all_results = []
    
    # Run experiments based on mode
    if args.mode in ["regular", "both"]:
        regular_results = run_regular_experiments(config, output_dir, param_names)
        all_results.extend(regular_results)
    
    if args.mode in ["dimensionless", "both"]:
        dimensionless_results = run_dimensionless_experiments(config, output_dir, param_names)
        all_results.extend(dimensionless_results)
    
    if args.mode == "normalized-eigenvalues":
        normalized_eigen_results = run_normalized_eigenvalue_experiments(config, output_dir, param_names)
        all_results.extend(normalized_eigen_results)
    
    # Generate reports and save results
    print("\nGenerating reports...")
    save_results_to_csv(all_results, output_dir)
    save_results_to_json(all_results, output_dir)
    
    print("\nUnified experiments completed!")
    print(f"Results directory: {output_dir}")
    
    # Print summary statistics
    print_console_summary(all_results)
    print_detailed_console_summary(all_results)


if __name__ == "__main__":
    main()