#!/usr/bin/env python3
"""
Multi-experiment KAN runner for NMR spectroscopy data.

This script runs multiple KAN experiments with different architectures and parameters,
comparing perturbation theory data vs precise eigenvalue data. It generates comprehensive
reports with symbolic expressions, training curves, and surface plots saved as PDF and CSV.

Usage:
    python run_many_experiments.py
"""

import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
import pandas as pd
import tqdm
import sympy as sp
from sympy import Expr

# Add the current directory to the path so we can import our custom modules
if "." not in sys.path:
    sys.path.append(".")

from nmrkan.models import KharKAN, _clean_expr, _fix_powers, _round_expr
from nmrkan.nmr import get_frequences_ordered
from nmrkan.sympy_torch import torchModelFromFormula
from report_utils import save_results_to_csv, print_console_summary, print_detailed_console_summary
from pdf_report import generate_pdf_report

warnings.filterwarnings('ignore', category=UserWarning)
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')


class ExperimentConfig:
    """Configuration class for experiments."""
    
    def __init__(self):
        # Model architectures to test - consistent mapping: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
        self.architectures = [
            (2, 4, 3),  # 2 inputs (deltaJ, deltaJ/Jintra), 4 hidden, 3 outputs
            (2, 6, 3),  # 2 inputs, 6 hidden, 3 outputs
            (2, 8, 3),  # 2 inputs, 8 hidden, 3 outputs
            (3, 4, 3),  # 3 inputs (deltaJ, deltaJ/Jintra, Jintra), 4 hidden, 3 outputs
            (3, 6, 3),  # 3 inputs, 6 hidden, 3 outputs
            (3, 8, 3),  # 3 inputs, 8 hidden, 3 outputs
            (
                4,
                4,
                3,
            ),  # 4 inputs (deltaJ, deltaJ/Jintra, Jintra, Jintra/deltaJ), 4 hidden, 3 outputs
            (4, 6, 3),  # 4 inputs, 6 hidden, 3 outputs
            (4, 8, 3),  # 4 inputs, 8 hidden, 3 outputs
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


def get_perturbation(Jintra: np.ndarray, deltaJ: np.ndarray) -> np.ndarray:
    """
    Perturbation theory approximation for NMR frequencies.
    
    Args:
        Jintra: Array of intramolecular coupling constants
        deltaJ: Array of intermolecular coupling differences
        
    Returns:
        Array of shape (B, 3) with perturbation theory frequencies
    """
    f1 = deltaJ / 2**0.5 + deltaJ * deltaJ / 8 / Jintra
    f2 = deltaJ / 2**0.5 - deltaJ * deltaJ / 8 / Jintra
    f0 = deltaJ * (2**0.5)
    return -np.stack((f2, f1, f0), axis=1).reshape(-1, 3)


def make_dataset_from_function(
    num_samples: int,
    func,
    min_Jintra: float = -32,
    max_Jintra: float = -5,
    min_deltaJ: float = -15,
    max_deltaJ: float = -0.1,
    ratio_threshold: float = 100,
    include_ratios: bool = True,
    input_mode: str = "3d",  # "2d", "3d", or "4d"
) -> Dict[str, torch.Tensor]:
    """
    Generate a dataset using a user-supplied function of two variables.

    Args:
        num_samples: Number of samples to generate
        func: Function that takes (Jintra, deltaJ) and returns labels
        min_Jintra, max_Jintra: Range for Jintra parameter
        min_deltaJ, max_deltaJ: Range for deltaJ parameter
        ratio_threshold: Maximum allowed ratio value (for filtering)
        include_ratios: Whether to include ratio features in inputs (for 3d/4d modes)
        input_mode: Type of input features - "2d" (deltaJ, deltaJ/Jintra), "3d" (Jintra, deltaJ, deltaJ/Jintra), "4d" (Jintra, deltaJ, deltaJ/Jintra, Jintra/deltaJ)

    Returns:
        Dictionary with 'train_input' and 'train_label' as torch tensors
    """
    jintra_list, deltaj_list, label_list = [], [], []
    samples_collected = 0
    
    while samples_collected < num_samples:
        # Generate more samples than needed to account for filtering
        batch_size = int((num_samples - samples_collected) * 2.0) + 200
        jintra = np.random.rand(batch_size) * (max_Jintra - min_Jintra) + min_Jintra
        deltaj = np.random.rand(batch_size) * (max_deltaJ - min_deltaJ) + min_deltaJ
        
        # Calculate ratios and filter by threshold
        ratiosX = deltaj / jintra
        ratiosY = jintra / deltaj
        # Keep only samples where absolute ratio is within threshold
        valid_mask = np.abs(ratiosX) <= ratio_threshold
        valid_mask &= np.abs(ratiosY) <= ratio_threshold
        jintra = jintra[valid_mask]
        deltaj = deltaj[valid_mask]
        
        # If no valid samples, continue to next iteration
        if len(jintra) == 0:
            continue
        
        # Apply function to get labels
        labels = func(jintra, deltaj)
        labels = np.array(labels, dtype=np.float32)
        
        # Take only what we need to avoid exceeding num_samples
        samples_to_take = min(len(jintra), num_samples - samples_collected)
        jintra_list.append(jintra[:samples_to_take])
        deltaj_list.append(deltaj[:samples_to_take])
        label_list.append(labels[:samples_to_take])
        samples_collected += samples_to_take
    
    # Concatenate and trim to exact size
    jintra = np.concatenate(jintra_list)[:num_samples]
    deltaj = np.concatenate(deltaj_list)[:num_samples]
    labels = np.concatenate(label_list)[:num_samples]
    ratioX = deltaj / jintra
    ratioY = jintra / deltaj

    # Prepare input features based on input_mode with consistent mapping:
    # x0 = deltaJ, x1 = deltaJ/Jintra, x2 = Jintra, x3 = Jintra/deltaJ
    if input_mode == "2d":
        # x0=deltaJ, x1=deltaJ/Jintra
        input_features = np.stack([deltaj, ratioX], axis=1)
    elif input_mode == "3d":
        # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra
        input_features = np.stack([deltaj, ratioX, jintra], axis=1)
    elif input_mode == "4d":
        # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
        input_features = np.stack([deltaj, ratioX, jintra, ratioY], axis=1)
    else:
        # Backward compatibility - use include_ratios parameter
        if include_ratios:
            input_features = np.stack([deltaj, ratioX, jintra, ratioY], axis=1)
        else:
            input_features = np.stack([deltaj, ratioX, jintra], axis=1)

    return {
        'train_input': torch.tensor(input_features).float(),
        'train_label': torch.tensor(labels).float()
    }


def fix_powers(expr: sp.Expr) -> sp.Expr:
    """Convert floating-point exponents that are actually integers to integer exponents."""
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sp.Pow) and isinstance(e.exp, sp.Float) and float(e.exp).is_integer(),
        lambda e: sp.Pow(e.base, int(e.exp))
    )


def round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    """Round all numbers in the expression to a given number of digits."""
    if expr is None:
        return None
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)


def train_kan_model(
    model: KharKAN,
    data: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    lr: float,
    l05_penalty: float,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Train a single KAN model.
    
    Returns:
        Tuple of (loss_history, mse_history, rel_err_history, final_mse)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction="none")  # Use reduction="none" like notebook
    
    inputs = data['train_input'].to(device)
    labels = data['train_label'].to(device)
    
    loss_history = []
    mse_history = []
    rel_err_history = []
    
    pbar = tqdm.tqdm(range(config.epochs), desc=f"Training LR={lr}, L05={l05_penalty}")
    
    for epoch in pbar:
        optimizer.zero_grad()
        preds = model(inputs)
        mse = criterion(preds, labels)  # This returns tensor of shape (batch_size, num_outputs)
        mse_norm = mse / (torch.abs(labels) + 1e-8)
        l05_loss = model.L05_loss()
        
        # Adaptive loss weighting - match notebook exactly
        if epoch < config.epochs // 2:
            loss = mse.mean() + l05_penalty * l05_loss + mse_norm.mean() * 100
        else:
            loss = mse.mean() + l05_penalty * l05_loss
            
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        mse_value = mse.mean().item()
        with torch.no_grad():
            rel_err = torch.mean(torch.abs(preds - labels) / (torch.abs(labels) + 1e-8)).item()
        
        loss_history.append(loss.item())
        mse_history.append(mse_value)
        rel_err_history.append(rel_err)
        
        if epoch % 100 == 0 or epoch == config.epochs - 1:
            pbar.set_postfix({
                "mse": f"{mse_value:.2e}",
                "l05": f"{(l05_penalty * l05_loss).item():.2e}",
                "rel_err": f"{rel_err:.2e}"
            })
        
        # Early stopping
        if rel_err < config.early_stop_threshold:
            pbar.close()
            break
    
    return loss_history, mse_history, rel_err_history, mse_value


def extract_symbolic_formulas(
    model: KharKAN, input_dim: int = 3, round_digits: int = 2
) -> Dict[str, Optional[sp.Expr]]:
    """Extract and clean symbolic formulas from trained model."""
    raw_formulas = model.symbolic_formula(round_digits=5)
    cleaned_formulas = {}

    x0, x1, x2, x3 = sp.symbols("x_0 x_1 x_2 x_3")
    j, dJ = sp.symbols("j dJ")  # Physical variable names

    for output_name in ["z_0", "z_1", "z_2"]:
        raw = raw_formulas.get(output_name, None)
        if raw is not None and isinstance(raw, sp.Expr):
            # Consistent mapping: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
            # Replace with meaningful variable names: dJ, dJ/j, j, j/dJ
            if input_dim == 2:
                # x0=deltaJ, x1=deltaJ/Jintra
                e = raw.subs({x0: dJ, x1: dJ / j})
            elif input_dim == 3:
                # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra
                e = raw.subs({x0: dJ, x1: dJ / j, x2: j})
            elif input_dim == 4:
                # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
                e = raw.subs({x0: dJ, x1: dJ / j, x2: j, x3: j / dJ})
            else:
                e = raw

            e = _fix_powers(e)
            e = _round_expr(e, 5)
            e = _clean_expr(e, eps=1e-5)
            e = _round_expr(e, round_digits)
            cleaned_formulas[output_name] = e
            cleaned_formulas[output_name] = e
        else:
            cleaned_formulas[output_name] = None

    return cleaned_formulas

def train_module_on_formula(formula: sp.Expr, inputs: torch.Tensor, labels: torch.Tensor, 
                           device: torch.device, epochs: int = 5000, lr: float = 0.001) -> Tuple[torch.nn.Module, float]:
    """
    Train a symbolic module created from a formula.
    
    Args:
        formula: SymPy expression to convert to PyTorch module
        inputs: Input tensor (batch_size, num_inputs)
        labels: Target tensor (batch_size, 1)
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Tuple of (trained_module, final_mse)
    """
    if formula is None:
        return None, float('inf')
    
    try:
        # Create PyTorch model from formula
        module = torchModelFromFormula(formula)
        module.to(device)
        
        # Setup training
        optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        # Prepare input dictionary for symbolic model
        input_dict = {}
        num_inputs = inputs.shape[1]
        for i in range(num_inputs):
            input_dict[f"x_{i}"] = inputs[:, i:i+1]
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = module(input_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate final MSE
        with torch.no_grad():
            final_outputs = module(input_dict)
            final_mse = criterion(final_outputs, labels).item()
        
        return module, final_mse
        
    except Exception as e:
        print(f"Error training symbolic module: {e}")
        return None, float('inf')


def finetune_symbolic_formulas(formulas: Dict[str, Optional[sp.Expr]], 
                              data: Dict[str, torch.Tensor], 
                              device: torch.device) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float], Dict[str, sp.Expr]]:
    """
    Fine-tune symbolic formulas using torchModelFromFormula.
    
    Args:
        formulas: Dictionary of symbolic formulas
        data: Dictionary with 'train_input' and 'train_label'
        device: Device to train on
        
    Returns:
        Tuple of (trained_modules, mse_scores, refined_formulas)
    """
    trained_modules = {}
    mse_scores = {}
    refined_formulas = {}
    
    inputs = data['train_input'].to(device)
    labels = data['train_label'].to(device)
    
    for i, (output_name, formula) in enumerate(formulas.items()):
        if formula is not None:
            print(f"Fine-tuning formula for {output_name}...")
            
            # Extract corresponding output labels
            output_labels = labels[:, i:i+1]
            
            # Train the symbolic module
            trained_module, final_mse = train_module_on_formula(
                formula, inputs, output_labels, device
            )
            
            if trained_module is not None:
                trained_modules[output_name] = trained_module
                mse_scores[output_name] = final_mse

                # Extract refined formula from trained module
                refined_formula = trained_module.to_sympy()
                refined_formula = round_expr(refined_formula, 3)
                refined_formulas[output_name] = refined_formula
                print(f"{output_name} - MSE after fine-tuning: {final_mse:.2e}")
            else:
                mse_scores[output_name] = float('inf')
                refined_formulas[output_name] = formula
        else:
            mse_scores[output_name] = float('inf')
            refined_formulas[output_name] = None
    
    return trained_modules, mse_scores, refined_formulas


def calculate_formula_complexity(formulas: Dict[str, Optional[sp.Expr]]) -> Dict[str, int]:
    """Calculate complexity (number of operations) for each formula."""
    complexities = {}
    for name, formula in formulas.items():
        if formula is not None:
            complexities[name] = sp.count_ops(formula)
        else:
            complexities[name] = float('inf')
    return complexities


def run_single_experiment(
    architecture: Tuple[int, ...],
    data_pert: Dict[str, torch.Tensor],
    data_eigen: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    lr: float,
    l05_penalty: float,
    experiment_id: str,
    params_name: str = ""
) -> Dict[str, Any]:
    """Run a single experiment configuration."""
    
    results = {
        'experiment_id': experiment_id,
        'architecture': architecture,
        'lr': lr,
        'l05_penalty': l05_penalty,
        'timestamp': time.time()
    }
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {experiment_id}")
    print(f"Architecture: {architecture}, LR: {lr}, L05: {l05_penalty}")
    print(f"Using parameter set: {params_name}")
    print(f"{'='*60}")
    
    # Run perturbation theory experiment
    print("\n--- Perturbation Theory Data ---")
    model_pert = KharKAN(architecture)

    loss_hist_pert, mse_hist_pert, rel_err_hist_pert, final_mse_pert = train_kan_model(
        model_pert, data_pert, config, lr, l05_penalty, config.device
    )

    formulas_pert = extract_symbolic_formulas(model_pert, architecture[0])
    complexity_pert = calculate_formula_complexity(formulas_pert)

    # Fine-tune symbolic formulas
    print("Fine-tuning symbolic formulas...")
    trained_modules_pert, mse_scores_pert, refined_formulas_pert = (
        finetune_symbolic_formulas(formulas_pert, data_pert, config.device)
    )

    results["perturbation"] = {
        "final_mse": final_mse_pert,
        "loss_history": loss_hist_pert,
        "mse_history": mse_hist_pert,
        "rel_err_history": rel_err_hist_pert,
        "formulas": formulas_pert,
        "refined_formulas": refined_formulas_pert,
        "symbolic_mse_scores": mse_scores_pert,
        "complexity": complexity_pert,
        "total_complexity": sum(
            c for c in complexity_pert.values() if c != float("inf")
        ),
    }

    print(f"Perturbation - Final MSE: {final_mse_pert:.2e}")
    for name, formula in formulas_pert.items():
        if formula is not None:
            print(f"{name}: {formula}")

    # Run eigenvalue experiment
    print("\n--- Eigenvalue Data ---")
    model_eigen = KharKAN(architecture)

    loss_hist_eigen, mse_hist_eigen, rel_err_hist_eigen, final_mse_eigen = (
        train_kan_model(model_eigen, data_eigen, config, lr, l05_penalty, config.device)
    )

    formulas_eigen = extract_symbolic_formulas(model_eigen, architecture[0])
    complexity_eigen = calculate_formula_complexity(formulas_eigen)

    # Fine-tune symbolic formulas
    print("Fine-tuning symbolic formulas...")
    trained_modules_eigen, mse_scores_eigen, refined_formulas_eigen = (
        finetune_symbolic_formulas(formulas_eigen, data_eigen, config.device)
    )

    results["eigenvalue"] = {
        "final_mse": final_mse_eigen,
        "loss_history": loss_hist_eigen,
        "mse_history": mse_hist_eigen,
        "rel_err_history": rel_err_hist_eigen,
        "formulas": formulas_eigen,
        "refined_formulas": refined_formulas_eigen,
        "symbolic_mse_scores": mse_scores_eigen,
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


def save_results_to_json(all_results, output_dir):
    """Save experiment results to JSON file."""
    
    # Convert torch tensors and sympy expressions to serializable format
    serializable_results = []
    
    for result in all_results:
        serializable_result = {
            'experiment_id': result['experiment_id'],
            'architecture': result['architecture'],
            'lr': result['lr'],
            'l05_penalty': result['l05_penalty'],
            'timestamp': result['timestamp']
        }
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result:
                data = result[data_type]
                if 'error' not in data:
                    serializable_data = {
                        'final_mse': data.get('final_mse'),
                        'total_complexity': data.get('total_complexity'),
                        'formulas': {},
                        'refined_formulas': {},
                        'symbolic_mse_scores': data.get('symbolic_mse_scores', {})
                    }
                    
                    # Convert sympy expressions to strings
                    for name, formula in data.get('formulas', {}).items():
                        if formula is not None:
                            serializable_data['formulas'][name] = str(formula)
                        else:
                            serializable_data['formulas'][name] = None
                    
                    # Convert refined formulas to strings
                    for name, formula in data.get('refined_formulas', {}).items():
                        if formula is not None:
                            serializable_data['refined_formulas'][name] = str(formula)
                        else:
                            serializable_data['refined_formulas'][name] = None
                    
                    serializable_result[data_type] = serializable_data
                else:
                    serializable_result[data_type] = data
        
        serializable_results.append(serializable_result)
    
    # Save to JSON
    json_path = output_dir / 'experiment_results.json'
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {json_path}")


def main():
    """Main function to run all experiments."""
    
    # Setup
    config = ExperimentConfig()
    output_dir = Path('experiment_results') / f"experiments_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {config.device}")
    print(f"Output directory: {output_dir}")
    
    # Iterate over all parameter sets for comprehensive testing
    param_names = [
        "working",
        "nonlinear",
        "realistic",
        "bidirectional",
        "realistic_truncated",
    ]
    all_datasets = {}
    
    print("\nGenerating datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Create datasets with different input dimensions for this parameter set
        datasets_2d = {}
        datasets_3d = {}
        datasets_4d = {}
        
        for arch in config.architectures:
            input_dim = arch[0]

            if input_dim == 2 and "2d" not in datasets_2d:
                # Generate 2-input datasets - first generate eigenvalue data
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

                # Generate perturbation data with same inputs as eigenvalue data
                # For 2D mode, we need to reconstruct Jintra from deltaJ and ratio
                # Since ratio = deltaJ/Jintra, then Jintra = deltaJ/ratio
                deltaJ_vals = data_eigen_2d["train_input"][
                    :, 0:1
                ].numpy()  # First column is deltaJ
                ratio_vals = data_eigen_2d["train_input"][
                    :, 1:2
                ].numpy()  # Second column is deltaJ/Jintra
                jintra_vals = deltaJ_vals / ratio_vals  # Reconstruct Jintra

                data_pert_2d = {
                    "train_input": data_eigen_2d["train_input"].clone(),
                    "train_label": torch.tensor(
                        get_perturbation(jintra_vals, deltaJ_vals)
                    ).float(),
                }

                print(f"2D Dataset shapes for {params_name}:")
                print(f"  Eigenvalue input: {data_eigen_2d['train_input'].shape}")
                print(f"  Eigenvalue label: {data_eigen_2d['train_label'].shape}")
                print(f"  Perturbation input: {data_pert_2d['train_input'].shape}")
                print(f"  Perturbation label: {data_pert_2d['train_label'].shape}")

                print(f"2D Data value ranges for {params_name}:")
                print(
                    f"  Eigenvalue label range: [{data_eigen_2d['train_label'].min():.2e}, {data_eigen_2d['train_label'].max():.2e}]"
                )
                print(
                    f"  Perturbation label range: [{data_pert_2d['train_label'].min():.2e}, {data_pert_2d['train_label'].max():.2e}]"
                )
                print(
                    f"  Eigenvalue label mean: {data_eigen_2d['train_label'].mean():.2e}"
                )
                print(
                    f"  Perturbation label mean: {data_pert_2d['train_label'].mean():.2e}"
                )

                datasets_2d = {"pert": data_pert_2d, "eigen": data_eigen_2d}

            elif input_dim == 3 and "3d" not in datasets_3d:
                # Generate 3-input datasets - first generate eigenvalue data
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
                
                # Generate perturbation data with same inputs as eigenvalue data
                # For 3D: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra
                data_pert_3d = {
                    "train_input": data_eigen_3d["train_input"].clone(),
                    "train_label": torch.tensor(
                        get_perturbation(
                            data_eigen_3d["train_input"][:, 2:3].numpy(),  # x2 = Jintra
                            data_eigen_3d["train_input"][:, 0:1].numpy(),  # x0 = deltaJ
                        )
                    ).float(),
                }
                
                print(f"3D Dataset shapes for {params_name}:")
                print(f"  Eigenvalue input: {data_eigen_3d['train_input'].shape}")
                print(f"  Eigenvalue label: {data_eigen_3d['train_label'].shape}")
                print(f"  Perturbation input: {data_pert_3d['train_input'].shape}")
                print(f"  Perturbation label: {data_pert_3d['train_label'].shape}")
                
                print(f"3D Data value ranges for {params_name}:")
                print(f"  Eigenvalue label range: [{data_eigen_3d['train_label'].min():.2e}, {data_eigen_3d['train_label'].max():.2e}]")
                print(f"  Perturbation label range: [{data_pert_3d['train_label'].min():.2e}, {data_pert_3d['train_label'].max():.2e}]")
                print(f"  Eigenvalue label mean: {data_eigen_3d['train_label'].mean():.2e}")
                print(f"  Perturbation label mean: {data_pert_3d['train_label'].mean():.2e}")
                
                datasets_3d = {'pert': data_pert_3d, 'eigen': data_eigen_3d}
                
            elif input_dim == 4 and '4d' not in datasets_4d:
                # Generate 4-input datasets - first generate eigenvalue data
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
                
                # Generate perturbation data with same inputs as eigenvalue data
                # For 4D: x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
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
    print("\nStarting experiments...")
    all_results = []
    experiment_count = 0
    total_experiments = len(param_names) * len(config.architectures) * len(config.learning_rates) * len(config.l05_penalties)
    
    for params_name in param_names:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR PARAMETER SET: {params_name.upper()}")
        print(f"{'='*80}")

        datasets_2d = all_datasets[params_name]["2d"]
        datasets_3d = all_datasets[params_name]['3d']
        datasets_4d = all_datasets[params_name]['4d']
        
        for architecture in config.architectures:
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    experiment_count += 1
                    experiment_id = f"{params_name}_arch_{architecture}_lr_{lr}_l05_{l05_penalty}"
                    
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
                            architecture, data_pert, data_eigen, config, lr, l05_penalty, experiment_id, params_name
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
                            'error': str(e)
                        })
    
    # Generate PDF report and save CSV results
    generate_pdf_report(all_results, all_datasets, config, output_dir, param_names)
    save_results_to_csv(all_results, output_dir)
    
    # Save results to JSON
    save_results_to_json(all_results, output_dir)
    
    print(f"\nExperiments completed!")
    print(f"Results directory: {output_dir}")
    
    # Print summary statistics using utility functions
    print_console_summary(all_results)
    print_detailed_console_summary(all_results)


if __name__ == "__main__":
    main()
