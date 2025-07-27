#!/usr/bin/env python3
"""
Multi-experiment KAN runner for NMR spectroscopy data.

This script runs multiple KAN experiments with different architectures and parameters,
codef extract_symbolic_formulas(model: KharKAN, round_digits: int = 3) -> Dict[str, Optional[sp.Expr]]:n theory data vs precise eigenvalue data. It generates comprehensive
reports with symbolic expressions, training curves, and surface plots saved as PDF.

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

warnings.filterwarnings('ignore', category=UserWarning)
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')


class ExperimentConfig:
    """Configuration class for experiments."""
    
    def __init__(self):
        # Model architectures to test
        self.architectures = [
            (3, 4, 3),      # 3 inputs (Jintra, deltaJ, ratio), 4 hidden, 3 outputs
            (3, 6, 3),      # 3 inputs, 6 hidden, 3 outputs
            (3, 8, 3),      # 3 inputs, 8 hidden, 3 outputs
            (4, 4, 3),      # 4 inputs (add ratio2), 4 hidden, 3 outputs
            (4, 6, 3),      # 4 inputs, 6 hidden, 3 outputs
            (4, 8, 3),      # 4 inputs, 8 hidden, 3 outputs
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
            "nonlinear" : {
                "MIN_JINTRA" : -32,
                "MAX_JINTRA" : -5,
                "MIN_DELTAJ" : -15,
                "MAX_DELTAJ" : -0.1,
                "RATIO_THRESHOLD" : 100,
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
                "RATIO_THRESHOLD": 100
            }
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
    return -np.stack((f2, f1, f0), axis=1)


def make_dataset_from_function(
    num_samples: int, 
    func,
    min_Jintra: float = -32, 
    max_Jintra: float = -5, 
    min_deltaJ: float = -15, 
    max_deltaJ: float = -0.1, 
    ratio_threshold: float = 100,
    include_ratios: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Generate a dataset using a user-supplied function of two variables.
    
    Args:
        num_samples: Number of samples to generate
        func: Function that takes (Jintra, deltaJ) and returns labels
        min_Jintra, max_Jintra: Range for Jintra parameter
        min_deltaJ, max_deltaJ: Range for deltaJ parameter  
        ratio_threshold: Maximum allowed ratio value (for filtering)
        include_ratios: Whether to include ratio features in inputs
        
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
    # Prepare input features
    if include_ratios:
        input_features = np.stack([jintra, deltaj, ratioX, ratioY], axis=1)
    else:
        input_features = np.stack([jintra, deltaj, ratioX], axis=1)
    
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


def extract_symbolic_formulas(model: KharKAN, round_digits: int = 2) -> Dict[str, Optional[sp.Expr]]:
    """Extract and clean symbolic formulas from trained model."""
    try:
        raw_formulas = model.symbolic_formula(round_digits=5)
        cleaned_formulas = {}
        
        x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
        
        for output_name in ['z_0', 'z_1', 'z_2']:
            raw = raw_formulas.get(output_name, None)
            if raw is not None and isinstance(raw, sp.Expr):
                # Substitute ratio features for interpretability
                e = raw.subs({x2: x1/x0, x3: x0/x1}).expand()
                e = _fix_powers(e)
                e = _round_expr(e, 5)
                e = _clean_expr(e, eps=1e-5)
                e = _round_expr(e, round_digits)
                cleaned_formulas[output_name] = e
            else:
                cleaned_formulas[output_name] = None
                
        return cleaned_formulas
    except Exception as e:
        print(f"Error extracting symbolic formulas: {e}")
        return {f'z_{i}': None for i in range(3)}


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
                try:
                    refined_formula = trained_module.to_sympy()
                    refined_formula = round_expr(refined_formula, 3)
                    refined_formulas[output_name] = refined_formula
                    print(f"{output_name} - MSE after fine-tuning: {final_mse:.2e}")
                except:
                    refined_formulas[output_name] = formula
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
    
    try:
        loss_hist_pert, mse_hist_pert, rel_err_hist_pert, final_mse_pert = train_kan_model(
            model_pert, data_pert, config, lr, l05_penalty, config.device
        )
        
        formulas_pert = extract_symbolic_formulas(model_pert)
        complexity_pert = calculate_formula_complexity(formulas_pert)
        
        # Fine-tune symbolic formulas
        print("Fine-tuning symbolic formulas...")
        trained_modules_pert, mse_scores_pert, refined_formulas_pert = finetune_symbolic_formulas(
            formulas_pert, data_pert, config.device
        )
        
        results['perturbation'] = {
            'final_mse': final_mse_pert,
            'loss_history': loss_hist_pert,
            'mse_history': mse_hist_pert,
            'rel_err_history': rel_err_hist_pert,
            'formulas': formulas_pert,
            'refined_formulas': refined_formulas_pert,
            'symbolic_mse_scores': mse_scores_pert,
            'complexity': complexity_pert,
            'total_complexity': sum(c for c in complexity_pert.values() if c != float('inf'))
        }
        
        print(f"Perturbation - Final MSE: {final_mse_pert:.2e}")
        for name, formula in formulas_pert.items():
            if formula is not None:
                print(f"{name}: {formula}")
                
    except Exception as e:
        print(f"Error in perturbation experiment: {e}")
        results['perturbation'] = {'error': str(e)}
    finally:
        del model_pert
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run eigenvalue experiment
    print("\n--- Eigenvalue Data ---")
    model_eigen = KharKAN(architecture)
    
    try:
        loss_hist_eigen, mse_hist_eigen, rel_err_hist_eigen, final_mse_eigen = train_kan_model(
            model_eigen, data_eigen, config, lr, l05_penalty, config.device
        )
        
        formulas_eigen = extract_symbolic_formulas(model_eigen)
        complexity_eigen = calculate_formula_complexity(formulas_eigen)
        
        # Fine-tune symbolic formulas
        print("Fine-tuning symbolic formulas...")
        trained_modules_eigen, mse_scores_eigen, refined_formulas_eigen = finetune_symbolic_formulas(
            formulas_eigen, data_eigen, config.device
        )
        
        results['eigenvalue'] = {
            'final_mse': final_mse_eigen,
            'loss_history': loss_hist_eigen,
            'mse_history': mse_hist_eigen,
            'rel_err_history': rel_err_hist_eigen,
            'formulas': formulas_eigen,
            'refined_formulas': refined_formulas_eigen,
            'symbolic_mse_scores': mse_scores_eigen,
            'complexity': complexity_eigen,
            'total_complexity': sum(c for c in complexity_eigen.values() if c != float('inf'))
        }
        
        print(f"Eigenvalue - Final MSE: {final_mse_eigen:.2e}")
        for name, formula in formulas_eigen.items():
            if formula is not None:
                print(f"{name}: {formula}")
                
    except Exception as e:
        print(f"Error in eigenvalue experiment: {e}")
        results['eigenvalue'] = {'error': str(e)}
    finally:
        del model_eigen
        torch.cuda.empty_cache()
        gc.collect()
    
    return results


def create_surface_plots(data_pert, data_eigen, fig, start_subplot=1):
    """Create 3D surface plots for the data."""
    
    try:
        # Extract data for plotting
        X_pert = data_pert['train_input'][:, 0].numpy()
        Y_pert = data_pert['train_input'][:, 1].numpy()
        Z_pert = data_pert['train_label'].numpy()
        
        X_eigen = data_eigen['train_input'][:, 0].numpy()
        Y_eigen = data_eigen['train_input'][:, 1].numpy()  
        Z_eigen = data_eigen['train_label'].numpy()
        
        print(f"Plotting data shapes:")
        print(f"  X_pert: {X_pert.shape}, Y_pert: {Y_pert.shape}, Z_pert: {Z_pert.shape}")
        print(f"  X_eigen: {X_eigen.shape}, Y_eigen: {Y_eigen.shape}, Z_eigen: {Z_eigen.shape}")
        
        # Sample subset for cleaner plots
        n_samples = min(2000, len(X_pert))
        indices = np.random.choice(len(X_pert), n_samples, replace=False)
        
        for i in range(min(3, Z_pert.shape[1] if len(Z_pert.shape) > 1 else 1)):  # Handle both 2D and 1D outputs
            # Perturbation data
            ax = fig.add_subplot(2, 3, start_subplot + i, projection='3d')
            
            if len(Z_pert.shape) > 1 and Z_pert.shape[1] > i:
                z_data_pert = Z_pert[indices, i]
            else:
                z_data_pert = Z_pert[indices] if i == 0 else np.zeros_like(indices, dtype=float)
                
            scatter = ax.scatter(X_pert[indices], Y_pert[indices], z_data_pert, 
                               c=z_data_pert, cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('Jintra')
            ax.set_ylabel('deltaJ')
            ax.set_zlabel(f'Frequency {i}')
            ax.set_title(f'Perturbation Theory - Output {i}')
            try:
                plt.colorbar(scatter, ax=ax, shrink=0.5)
            except Exception as e:
                print(f"Colorbar error for perturbation plot {i}: {e}")
            
            # Eigenvalue data
            ax = fig.add_subplot(2, 3, start_subplot + i + 3, projection='3d')
            
            if len(Z_eigen.shape) > 1 and Z_eigen.shape[1] > i:
                z_data_eigen = Z_eigen[indices, i]
            else:
                z_data_eigen = Z_eigen[indices] if i == 0 else np.zeros_like(indices, dtype=float)
                
            scatter = ax.scatter(X_eigen[indices], Y_eigen[indices], z_data_eigen, 
                               c=z_data_eigen, cmap='plasma', s=1, alpha=0.6)
            ax.set_xlabel('Jintra')
            ax.set_ylabel('deltaJ')
            ax.set_zlabel(f'Frequency {i}')
            ax.set_title(f'Eigenvalue - Output {i}')
            try:
                plt.colorbar(scatter, ax=ax, shrink=0.5)
            except Exception as e:
                print(f"Colorbar error for eigenvalue plot {i}: {e}")
                
    except Exception as e:
        print(f"Error creating surface plots: {e}")
        # Create a simple text plot instead
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Error creating plots: {str(e)}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')


def plot_training_curves(results, fig, start_subplot=1):
    """Plot training curves for loss, MSE, and relative error."""
    
    if 'perturbation' not in results or 'eigenvalue' not in results:
        return
        
    pert_data = results['perturbation']
    eigen_data = results['eigenvalue']
    
    if 'error' in pert_data or 'error' in eigen_data:
        return
    
    # Loss curves
    ax1 = fig.add_subplot(2, 3, start_subplot)
    if 'loss_history' in pert_data:
        ax1.plot(pert_data['loss_history'], label='Perturbation', alpha=0.8)
    if 'loss_history' in eigen_data:
        ax1.plot(eigen_data['loss_history'], label='Eigenvalue', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE curves
    ax2 = fig.add_subplot(2, 3, start_subplot + 1)
    if 'mse_history' in pert_data:
        ax2.plot(pert_data['mse_history'], label='Perturbation', alpha=0.8)
    if 'mse_history' in eigen_data:
        ax2.plot(eigen_data['mse_history'], label='Eigenvalue', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.set_title('Mean Squared Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Relative error curves
    ax3 = fig.add_subplot(2, 3, start_subplot + 2)
    if 'rel_err_history' in pert_data:
        ax3.plot(pert_data['rel_err_history'], label='Perturbation', alpha=0.8)
    if 'rel_err_history' in eigen_data:
        ax3.plot(eigen_data['rel_err_history'], label='Eigenvalue', alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Relative Error')
    ax3.set_title('Relative Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


def create_experiment_summary_page(all_results, fig):
    """Create a summary page with experiment comparison."""
    
    # Clear the figure
    fig.clear()
    
    # Create summary statistics
    summary_data = []
    for result in all_results:
        if 'perturbation' in result and 'eigenvalue' in result:
            pert = result['perturbation']
            eigen = result['eigenvalue']
            
            if 'error' not in pert and 'error' not in eigen:
                summary_data.append({
                    'Architecture': str(result['architecture']),
                    'LR': result['lr'],
                    'L05': result['l05_penalty'],
                    'Pert_MSE': pert.get('final_mse', float('inf')),
                    'Eigen_MSE': eigen.get('final_mse', float('inf')),
                    'Pert_Complexity': pert.get('total_complexity', float('inf')),
                    'Eigen_Complexity': eigen.get('total_complexity', float('inf'))
                })
    
    if not summary_data:
        return
        
    df = pd.DataFrame(summary_data)
    
    # MSE comparison plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df['Pert_MSE'], df['Eigen_MSE'], alpha=0.7, s=50)
    ax1.set_xlabel('Perturbation MSE')
    ax1.set_ylabel('Eigenvalue MSE')
    ax1.set_title('MSE Comparison: Perturbation vs Eigenvalue')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot([df['Pert_MSE'].min(), df['Pert_MSE'].max()], 
             [df['Pert_MSE'].min(), df['Pert_MSE'].max()], 'r--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Complexity comparison plot
    ax2 = fig.add_subplot(2, 2, 2)
    valid_complexity = (df['Pert_Complexity'] != float('inf')) & (df['Eigen_Complexity'] != float('inf'))
    if valid_complexity.any():
        ax2.scatter(df.loc[valid_complexity, 'Pert_Complexity'], 
                   df.loc[valid_complexity, 'Eigen_Complexity'], alpha=0.7, s=50)
        ax2.set_xlabel('Perturbation Complexity')
        ax2.set_ylabel('Eigenvalue Complexity')
        ax2.set_title('Formula Complexity Comparison')
        ax2.grid(True, alpha=0.3)
    
    # Best architectures table
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('tight')
    ax3.axis('off')
    
    # Sort by MSE and get top 5
    df_sorted = df.nsmallest(5, 'Pert_MSE')
    table_data = df_sorted[['Architecture', 'LR', 'L05', 'Pert_MSE', 'Eigen_MSE']].round(6)
    
    table = ax3.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    ax3.set_title('Top 5 Architectures (by Perturbation MSE)')
    
    # Performance heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Create pivot table for heatmap
    if len(df) > 1 and SEABORN_AVAILABLE:
        pivot_data = df.pivot_table(values='Pert_MSE', index='Architecture', columns='LR', aggfunc='mean')
        if not pivot_data.empty:
            sns.heatmap(np.log10(pivot_data), annot=True, fmt='.1f', cmap='viridis_r', ax=ax4)
            ax4.set_title('Log10(MSE) Heatmap by Architecture and LR')
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('Architecture')
    else:
        ax4.text(0.5, 0.5, 'Heatmap requires seaborn\nand multiple experiments', 
                ha='center', va='center', transform=ax4.transAxes)


def create_formula_table_page(all_results, fig):
    """Create a comprehensive table of all formulas discovered."""
    
    fig.clear()
    fig.suptitle('Complete Formula Summary', fontsize=16, fontweight='bold')
    
    # Collect all formulas
    formula_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        arch = str(result['architecture'])
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                data = result[data_type]
                
                # Original formulas
                formulas = data.get('formulas', {})
                refined_formulas = data.get('refined_formulas', {})
                symbolic_mse = data.get('symbolic_mse_scores', {})
                
                for output_name in ['z_0', 'z_1', 'z_2']:
                    original_formula = formulas.get(output_name)
                    refined_formula = refined_formulas.get(output_name)
                    mse_score = symbolic_mse.get(output_name, float('inf'))
                    
                    if original_formula is not None:
                        formula_data.append({
                            'Experiment': exp_id,
                            'Architecture': arch,
                            'Data_Type': data_type.capitalize(),
                            'Output': output_name,
                            'Original_Formula': str(original_formula)[:100] + ('...' if len(str(original_formula)) > 100 else ''),
                            'Refined_Formula': str(refined_formula)[:100] + ('...' if len(str(refined_formula)) > 100 else '') if refined_formula else 'None',
                            'Symbolic_MSE': f"{mse_score:.2e}" if mse_score != float('inf') else 'Inf',
                            'Complexity': sp.count_ops(original_formula) if original_formula else 'N/A'
                        })
    
    if not formula_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No formulas found in experiments', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create table
    df = pd.DataFrame(formula_data)
    
    # Sort by symbolic MSE (best first)
    df_sorted = df.sort_values('Symbolic_MSE')
    
    # Split into multiple tables if too many rows
    rows_per_page = 20
    num_pages = (len(df_sorted) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df_sorted))
        page_data = df_sorted.iloc[start_idx:end_idx]
        
        ax = fig.add_subplot(num_pages, 1, page + 1)
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with appropriate columns
        display_cols = ['Experiment', 'Architecture', 'Data_Type', 'Output', 'Symbolic_MSE', 'Complexity']
        table_data = page_data[display_cols]
        
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.2, 1.5)
        
        if page == 0:
            ax.set_title(f'Formula Results (Page {page+1}/{num_pages}) - Sorted by Symbolic MSE', pad=20)
        else:
            ax.set_title(f'Formula Results (Page {page+1}/{num_pages})', pad=20)


def create_comprehensive_summary_table(all_results, fig):
    """Create a comprehensive summary table with MSE, formulas, and complexity for each parameter set and architecture."""
    
    fig.clear()
    fig.suptitle('Comprehensive Results Summary', fontsize=16, fontweight='bold')
    
    # Collect data for summary table
    summary_data = []
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0]  # Extract parameter set name
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D"
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                data = result[data_type]
                
                # Extract MSE
                final_mse = data.get('final_mse', float('inf'))
                
                # Extract formulas and complexities
                formulas = data.get('formulas', {})
                refined_formulas = data.get('refined_formulas', {})
                symbolic_mse_scores = data.get('symbolic_mse_scores', {})
                
                for output_name in ['z_0', 'z_1', 'z_2']:
                    original_formula = formulas.get(output_name)
                    refined_formula = refined_formulas.get(output_name)
                    symbolic_mse = symbolic_mse_scores.get(output_name, float('inf'))
                    
                    # Calculate complexities
                    orig_complexity = sp.count_ops(original_formula) if original_formula else 0
                    refined_complexity = sp.count_ops(refined_formula) if refined_formula else 0
                    
                    # Format formulas for display (truncate if too long)
                    orig_formula_str = str(original_formula)[:50] + ('...' if len(str(original_formula)) > 50 else '') if original_formula else 'None'
                    refined_formula_str = str(refined_formula)[:50] + ('...' if len(str(refined_formula)) > 50 else '') if refined_formula else 'None'
                    
                    summary_data.append({
                        'Param_Set': params_name.upper(),
                        'Input_Dim': input_dim,
                        'Architecture': str(architecture),
                        'Data_Type': data_type.capitalize(),
                        'Output': output_name,
                        'Neural_MSE': f"{final_mse:.2e}",
                        'Symbolic_MSE': f"{symbolic_mse:.2e}" if symbolic_mse != float('inf') else 'Inf',
                        'Original_Formula': orig_formula_str,
                        'Original_Complexity': orig_complexity,
                        'Refined_Formula': refined_formula_str,
                        'Refined_Complexity': refined_complexity
                    })
    
    if not summary_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No results found for summary table', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create DataFrame and sort by parameter set, input dimension, and MSE
    df = pd.DataFrame(summary_data)
    df_sorted = df.sort_values(['Param_Set', 'Input_Dim', 'Neural_MSE'])
    
    # Split into multiple pages if too many rows
    rows_per_page = 25
    num_pages = (len(df_sorted) + rows_per_page - 1) // rows_per_page
    
    for page in range(num_pages):
        start_idx = page * rows_per_page
        end_idx = min((page + 1) * rows_per_page, len(df_sorted))
        page_data = df_sorted.iloc[start_idx:end_idx]
        
        ax = fig.add_subplot(num_pages, 1, page + 1)
        ax.axis('tight')
        ax.axis('off')
        
        # Select columns for display
        display_cols = ['Param_Set', 'Input_Dim', 'Data_Type', 'Output', 'Neural_MSE', 
                       'Symbolic_MSE', 'Original_Complexity', 'Refined_Complexity']
        table_data = page_data[display_cols]
        
        # Create table
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 1.8)
        
        # Color-code by parameter set
        param_colors = {'WORKING': '#FFE6E6', 'NONLINEAR': '#E6F3FF', 
                       'REALISTIC': '#E6FFE6', 'BIDIRECTIONAL': '#FFFFE6'}
        
        for i, (idx, row) in enumerate(page_data.iterrows()):
            param_set = row['Param_Set']
            if param_set in param_colors:
                for j in range(len(display_cols)):
                    table[(i+1, j)].set_facecolor(param_colors[param_set])
        
        if page == 0:
            ax.set_title(f'Summary Table (Page {page+1}/{num_pages}) - All Parameter Sets and Architectures', pad=20)
        else:
            ax.set_title(f'Summary Table (Page {page+1}/{num_pages})', pad=20)


def create_best_results_summary(all_results, fig):
    """Create a summary of best results for each parameter set and architecture combination."""
    
    fig.clear()
    fig.suptitle('Best Results Summary by Parameter Set and Architecture', fontsize=16, fontweight='bold')
    
    # Group results by parameter set and input dimension
    grouped_results = {}
    
    for result in all_results:
        if 'error' in result:
            continue
            
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0]
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D"
        
        key = f"{params_name}_{input_dim}"
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Find best results for each group
    best_results_data = []
    
    for group_key, group_results in grouped_results.items():
        params_name, input_dim = group_key.split('_')
        
        for data_type in ['perturbation', 'eigenvalue']:
            # Find best result by Neural MSE
            valid_results = [r for r in group_results 
                           if data_type in r and 'error' not in r[data_type]]
            
            if valid_results:
                best_result = min(valid_results, 
                                key=lambda x: x[data_type].get('final_mse', float('inf')))
                
                data = best_result[data_type]
                final_mse = data.get('final_mse', float('inf'))
                total_complexity_orig = data.get('total_complexity', 0)
                
                # Calculate total refined complexity
                refined_formulas = data.get('refined_formulas', {})
                total_complexity_refined = sum(sp.count_ops(f) for f in refined_formulas.values() if f is not None)
                
                # Get best symbolic MSE
                symbolic_mses = data.get('symbolic_mse_scores', {})
                best_symbolic_mse = min(symbolic_mses.values()) if symbolic_mses else float('inf')
                
                # Count successful formulas
                successful_formulas = sum(1 for f in data.get('formulas', {}).values() if f is not None)
                successful_refined = sum(1 for f in refined_formulas.values() if f is not None)
                
                best_results_data.append({
                    'Parameter_Set': params_name.upper(),
                    'Input_Dim': input_dim,
                    'Data_Type': data_type.capitalize(),
                    'Architecture': str(best_result['architecture']),
                    'Neural_MSE': f"{final_mse:.2e}",
                    'Best_Symbolic_MSE': f"{best_symbolic_mse:.2e}" if best_symbolic_mse != float('inf') else 'Inf',
                    'Formulas_Found': f"{successful_formulas}/3",
                    'Refined_Formulas': f"{successful_refined}/3",
                    'Original_Complexity': total_complexity_orig,
                    'Refined_Complexity': total_complexity_refined,
                    'Experiment_ID': best_result['experiment_id']
                })
    
    if not best_results_data:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No best results found', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        return
    
    # Create DataFrame and sort
    df = pd.DataFrame(best_results_data)
    df_sorted = df.sort_values(['Parameter_Set', 'Input_Dim', 'Data_Type'])
    
    # Create table
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Select columns for display
    display_cols = ['Parameter_Set', 'Input_Dim', 'Data_Type', 'Neural_MSE', 
                   'Best_Symbolic_MSE', 'Formulas_Found', 'Original_Complexity', 'Refined_Complexity']
    table_data = df_sorted[display_cols]
    
    # Create table
    table = ax.table(cellText=table_data.values,
                    colLabels=table_data.columns,
                    cellLoc='center',
                    loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    # Color-code by parameter set
    param_colors = {'WORKING': '#FFE6E6', 'NONLINEAR': '#E6F3FF', 
                   'REALISTIC': '#E6FFE6', 'BIDIRECTIONAL': '#FFFFE6'}
    
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        param_set = row['Parameter_Set']
        if param_set in param_colors:
            for j in range(len(display_cols)):
                table[(i+1, j)].set_facecolor(param_colors[param_set])
    
    ax.set_title('Best Performance for Each Parameter Set and Architecture Combination', pad=20)


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
    param_names = ["working", "nonlinear", "realistic", "bidirectional"]
    all_datasets = {}
    
    print("\nGenerating datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Create datasets with different input dimensions for this parameter set
        datasets_3d = {}
        datasets_4d = {}
    
        # Create datasets with different input dimensions for this parameter set
        datasets_3d = {}
        datasets_4d = {}
        
        for arch in config.architectures:
            input_dim = arch[0]
            
            if input_dim == 3 and '3d' not in datasets_3d:
                # Generate 3-input datasets - first generate eigenvalue data
                data_eigen_3d = make_dataset_from_function(
                    config.num_samples, get_frequences_ordered,
                    min_Jintra=params["MIN_JINTRA"], max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"], max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"], include_ratios=False
                )
                
                # Generate perturbation data with same inputs as eigenvalue data
                data_pert_3d = {
                    'train_input': data_eigen_3d['train_input'].clone(),
                    'train_label': torch.tensor(get_perturbation(
                        data_eigen_3d['train_input'][:, 0:1].numpy(), 
                        data_eigen_3d['train_input'][:, 1:2].numpy()
                    )).float()
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
                    config.num_samples, get_frequences_ordered,
                    min_Jintra=params["MIN_JINTRA"], max_Jintra=params["MAX_JINTRA"],
                    min_deltaJ=params["MIN_DELTAJ"], max_deltaJ=params["MAX_DELTAJ"],
                    ratio_threshold=params["RATIO_THRESHOLD"], include_ratios=True
                )
                
                # Generate perturbation data with same inputs as eigenvalue data
                data_pert_4d = {
                    'train_input': data_eigen_4d['train_input'].clone(),
                    'train_label': torch.tensor(get_perturbation(
                        data_eigen_4d['train_input'][:, 0:1].numpy(), 
                        data_eigen_4d['train_input'][:, 1:2].numpy()
                    )).float()
                }
                
                # Add second ratio feature (reciprocal of first ratio) to both datasets
                ratio2_pert = data_pert_4d['train_input'][:, 0:1] / data_pert_4d['train_input'][:, 1:2]
                ratio2_eigen = data_eigen_4d['train_input'][:, 0:1] / data_eigen_4d['train_input'][:, 1:2]
                
                data_pert_4d['train_input'] = torch.cat([data_pert_4d['train_input'], ratio2_pert], dim=1)
                data_eigen_4d['train_input'] = torch.cat([data_eigen_4d['train_input'], ratio2_eigen], dim=1)
                
                datasets_4d = {'pert': data_pert_4d, 'eigen': data_eigen_4d}
        
        # Store datasets for this parameter set
        all_datasets[params_name] = {
            '3d': datasets_3d,
            '4d': datasets_4d
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
        
        datasets_3d = all_datasets[params_name]['3d']
        datasets_4d = all_datasets[params_name]['4d']
        
        for architecture in config.architectures:
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    experiment_count += 1
                    experiment_id = f"{params_name}_arch_{architecture}_lr_{lr}_l05_{l05_penalty}"
                    
                    print(f"\nExperiment {experiment_count}/{total_experiments}: {experiment_id}")
                    
                    # Select appropriate dataset based on input dimension
                    if architecture[0] == 3:
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
    
    # Generate PDF report
    print("\nGenerating PDF report...")
    pdf_path = output_dir / 'experiment_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('KAN NMR Spectroscopy Experiments Report', fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        report_text = f"""
        Comprehensive KAN Experiments for NMR Spectroscopy
        
        Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
        
        Experiment Configuration:
        • Parameter sets tested: {len(param_names)} ({param_names})
        • Architectures tested: {len(config.architectures)} ({config.architectures})
        • Learning rates: {config.learning_rates}
        • L05 penalties: {config.l05_penalties}
        • Training epochs: {config.epochs}
        • Number of samples: {config.num_samples}
        • Device: {config.device}
        
        Parameter Set Ranges:
        """
        
        for pname in param_names:
            pparams = config.data_params[pname]
            report_text += f"""
        • {pname.upper()}:
          - Jintra: [{pparams["MIN_JINTRA"]}, {pparams["MAX_JINTRA"]}]
          - deltaJ: [{pparams["MIN_DELTAJ"]}, {pparams["MAX_DELTAJ"]}]
        """
        
        report_text += f"""
        Total experiments completed: {len([r for r in all_results if 'error' not in r])} / {total_experiments}
        """
        
        ax.text(0.1, 0.9, report_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Data visualization pages for each parameter set
        for params_name in param_names:
            datasets_3d = all_datasets[params_name]['3d']
            datasets_4d = all_datasets[params_name]['4d']
            
            if datasets_3d:
                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f'3D Data Surfaces - {params_name.upper()} Parameter Set - 3 Input Architecture', fontsize=14, fontweight='bold')
                create_surface_plots(datasets_3d['pert'], datasets_3d['eigen'], fig)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            if datasets_4d:
                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f'3D Data Surfaces - {params_name.upper()} Parameter Set - 4 Input Architecture', fontsize=14, fontweight='bold')
                create_surface_plots(datasets_4d['pert'], datasets_4d['eigen'], fig)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # Summary page
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Experiment Summary and Comparison', fontsize=14, fontweight='bold')
        create_experiment_summary_page(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Formula table page
        fig = plt.figure(figsize=(15, 12))
        create_formula_table_page(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Comprehensive summary table
        fig = plt.figure(figsize=(20, 15))
        create_comprehensive_summary_table(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Best results summary
        fig = plt.figure(figsize=(15, 10))
        create_best_results_summary(all_results, fig)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Individual experiment pages
        for result in all_results:
            if 'error' in result:
                continue
                
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle(f"Experiment: {result['experiment_id']}", fontsize=14, fontweight='bold')
            
            # Plot training curves
            plot_training_curves(result, fig, 1)
            
            # Add formula text
            ax_text = fig.add_subplot(2, 3, 4)
            ax_text.axis('off')
            
            formula_text = f"Architecture: {result['architecture']}\n"
            formula_text += f"LR: {result['lr']}, L05: {result['l05_penalty']}\n\n"
            
            if 'perturbation' in result and 'error' not in result['perturbation']:
                formula_text += "PERTURBATION THEORY FORMULAS:\n"
                formula_text += "Original Formulas:\n"
                for name, formula in result['perturbation'].get('formulas', {}).items():
                    if formula is not None:
                        formula_text += f"  {name}: {formula}\n"
                    else:
                        formula_text += f"  {name}: None\n"
                
                formula_text += "Refined Formulas (after fine-tuning):\n"
                for name, formula in result['perturbation'].get('refined_formulas', {}).items():
                    if formula is not None:
                        formula_text += f"  {name}: {formula}\n"
                    else:
                        formula_text += f"  {name}: None\n"
                
                formula_text += "Symbolic MSE Scores:\n"
                for name, mse in result['perturbation'].get('symbolic_mse_scores', {}).items():
                    formula_text += f"  {name}: {mse:.2e}\n"
                
                formula_text += f"Total Complexity: {result['perturbation'].get('total_complexity', 'N/A')}\n"
                formula_text += f"Final Neural MSE: {result['perturbation'].get('final_mse', 'N/A'):.2e}\n\n"
            
            if 'eigenvalue' in result and 'error' not in result['eigenvalue']:
                formula_text += "EIGENVALUE FORMULAS:\n"
                formula_text += "Original Formulas:\n"
                for name, formula in result['eigenvalue'].get('formulas', {}).items():
                    if formula is not None:
                        formula_text += f"  {name}: {formula}\n"
                    else:
                        formula_text += f"  {name}: None\n"
                
                formula_text += "Refined Formulas (after fine-tuning):\n"
                for name, formula in result['eigenvalue'].get('refined_formulas', {}).items():
                    if formula is not None:
                        formula_text += f"  {name}: {formula}\n"
                    else:
                        formula_text += f"  {name}: None\n"
                
                formula_text += "Symbolic MSE Scores:\n"
                for name, mse in result['eigenvalue'].get('symbolic_mse_scores', {}).items():
                    formula_text += f"  {name}: {mse:.2e}\n"
                
                formula_text += f"Total Complexity: {result['eigenvalue'].get('total_complexity', 'N/A')}\n"
                formula_text += f"Final Neural MSE: {result['eigenvalue'].get('final_mse', 'N/A'):.2e}\n"
            
            ax_text.text(0.05, 0.95, formula_text, transform=ax_text.transAxes,
                        fontsize=8, verticalalignment='top', fontfamily='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    # Save results to JSON
    save_results_to_json(all_results, output_dir)
    
    print(f"\nExperiments completed!")
    print(f"PDF report saved to: {pdf_path}")
    print(f"Results directory: {output_dir}")
    
    # Print summary statistics
    successful_results = [r for r in all_results if 'error' not in r]
    print(f"\nSummary: {len(successful_results)}/{total_experiments} experiments completed successfully")
    
    # Group results by parameter set
    results_by_params = {}
    for result in successful_results:
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0]  # Extract parameter set name from experiment ID
        if params_name not in results_by_params:
            results_by_params[params_name] = []
        results_by_params[params_name].append(result)
    
    print(f"\nResults by Parameter Set:")
    for params_name, param_results in results_by_params.items():
        print(f"\n{params_name.upper()} Parameter Set:")
        print(f"  Completed experiments: {len(param_results)}")
        
        if param_results:
            best_pert = min(param_results, 
                           key=lambda x: x.get('perturbation', {}).get('final_mse', float('inf')))
            best_eigen = min(param_results,
                            key=lambda x: x.get('eigenvalue', {}).get('final_mse', float('inf')))
            
            print(f"  Best Perturbation MSE: {best_pert.get('perturbation', {}).get('final_mse', 'N/A'):.2e} ({best_pert['experiment_id']})")
            print(f"  Best Eigenvalue MSE: {best_eigen.get('eigenvalue', {}).get('final_mse', 'N/A'):.2e} ({best_eigen['experiment_id']})")
    
    if successful_results:
        overall_best_pert = min(successful_results, 
                               key=lambda x: x.get('perturbation', {}).get('final_mse', float('inf')))
        overall_best_eigen = min(successful_results,
                                key=lambda x: x.get('eigenvalue', {}).get('final_mse', float('inf')))
        
        print(f"\nOverall Best Results:")
        print(f"  Best Perturbation Theory Result:")
        print(f"    Experiment: {overall_best_pert['experiment_id']}")
        print(f"    MSE: {overall_best_pert.get('perturbation', {}).get('final_mse', 'N/A'):.2e}")
        
        print(f"  Best Eigenvalue Result:")
        print(f"    Experiment: {overall_best_eigen['experiment_id']}")
        print(f"    MSE: {overall_best_eigen.get('eigenvalue', {}).get('final_mse', 'N/A'):.2e}")
    
    # Print comprehensive summary table to console
    print(f"\n{'='*100}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Param Set':<12} {'Input':<5} {'Data Type':<12} {'Neural MSE':<12} {'Best Sym MSE':<12} {'Formulas':<8} {'Orig Compl':<10} {'Ref Compl':<10}")
    print("-" * 100)
    
    # Group results by parameter set and input dimension for summary
    summary_stats = {}
    
    for result in successful_results:
        exp_id = result['experiment_id']
        params_name = exp_id.split('_')[0]
        architecture = result['architecture']
        input_dim = f"{architecture[0]}D"
        
        key = f"{params_name}_{input_dim}"
        if key not in summary_stats:
            summary_stats[key] = {'perturbation': [], 'eigenvalue': []}
        
        for data_type in ['perturbation', 'eigenvalue']:
            if data_type in result and 'error' not in result[data_type]:
                summary_stats[key][data_type].append(result[data_type])
    
    # Print summary for each parameter set and dimension
    for key, data in summary_stats.items():
        params_name, input_dim = key.split('_')
        
        for data_type, results_list in data.items():
            if results_list:
                # Find best neural MSE
                best_neural_mse = min(r.get('final_mse', float('inf')) for r in results_list)
                
                # Find best symbolic MSE across all outputs
                all_symbolic_mses = []
                total_formulas = 0
                total_orig_complexity = 0
                total_refined_complexity = 0
                
                for r in results_list:
                    symbolic_mses = r.get('symbolic_mse_scores', {})
                    all_symbolic_mses.extend([mse for mse in symbolic_mses.values() if mse != float('inf')])
                    
                    formulas = r.get('formulas', {})
                    total_formulas += sum(1 for f in formulas.values() if f is not None)
                    
                    total_orig_complexity += r.get('total_complexity', 0)
                    
                    refined_formulas = r.get('refined_formulas', {})
                    total_refined_complexity += sum(sp.count_ops(f) for f in refined_formulas.values() if f is not None)
                
                best_symbolic_mse = min(all_symbolic_mses) if all_symbolic_mses else float('inf')
                avg_orig_complexity = total_orig_complexity / len(results_list) if results_list else 0
                avg_refined_complexity = total_refined_complexity / len(results_list) if results_list else 0
                total_possible_formulas = len(results_list) * 3  # 3 outputs per experiment
                
                print(f"{params_name.upper():<12} {input_dim:<5} {data_type.capitalize():<12} "
                      f"{best_neural_mse:<12.2e} {best_symbolic_mse:<12.2e} "
                      f"{total_formulas}/{total_possible_formulas:<8} {avg_orig_complexity:<10.1f} {avg_refined_complexity:<10.1f}")
    
    print("-" * 100)
    print("Legend: Param Set = Parameter Set, Input = Input Dimension, Neural MSE = Best Neural Network MSE")
    print("        Best Sym MSE = Best Symbolic MSE, Formulas = Successful/Total, Compl = Average Complexity")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
