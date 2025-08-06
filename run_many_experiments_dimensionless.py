#!/usr/bin/env python3
"""
Dimensionless multi-experiment KAN runner for NMR spectroscopy data.

This script runs multiple KAN experiments using dimensionless variables as inputs:
- 1D input: dJ/J (dimensionless ratio)
- 2D input: dJ/J and J/dJ (dimensionless ratios)

The model predicts dimensionless outputs Z0/J, Z1/J, Z2/J and then scales them back
to physical units by multiplying by J to get Z0, Z1, Z2.

Usage:
    python create_many_experiments_dimensionless.py
"""

import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
try:
    import seaborn as sns  # noqa: F401
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
import tqdm
import sympy as sp

# Add the current directory to the path so we can import our custom modules
if "." not in sys.path:
    sys.path.append(".")

from nmrkan.models import KharKAN, _clean_expr, _fix_powers, _round_expr
from nmrkan.nmr import get_frequences_ordered
from report_utils import save_results_to_csv, print_console_summary, print_detailed_console_summary
from pdf_report import generate_pdf_report

warnings.filterwarnings('ignore', category=UserWarning)
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')


class DimensionlessExperimentConfig:
    """Configuration class for dimensionless experiments."""
    
    def __init__(self):
        # Model architectures to test - only 1D and 2D inputs for dimensionless variables
        self.architectures = [
            (1, 4, 3),  # 1 input (dJ/J), 4 hidden, 3 outputs (Z0/J, Z1/J, Z2/J)
            (1, 6, 3),  # 1 input (dJ/J), 6 hidden, 3 outputs
            (1, 8, 3),  # 1 input (dJ/J), 8 hidden, 3 outputs
            (2, 4, 3),  # 2 inputs (dJ/J, J/dJ), 4 hidden, 3 outputs
            (2, 6, 3),  # 2 inputs (dJ/J, J/dJ), 6 hidden, 3 outputs
            (2, 8, 3),  # 2 inputs (dJ/J, J/dJ), 8 hidden, 3 outputs
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


def get_perturbation_dimensionless(dJ_over_J: np.ndarray) -> np.ndarray:
    """
    Perturbation theory approximation for dimensionless NMR frequencies (Z/J).
    
    Args:
        dJ_over_J: Array of dimensionless coupling ratios (deltaJ/Jintra)
        
    Returns:
        Array of shape (B, 3) with dimensionless perturbation theory frequencies (Z/J)
    """
    # Original perturbation formulas:
    # f1 = deltaJ / 2**0.5 + deltaJ * deltaJ / 8 / Jintra
    # f2 = deltaJ / 2**0.5 - deltaJ * deltaJ / 8 / Jintra  
    # f0 = deltaJ * (2**0.5)
    
    # Divide by J to get dimensionless:
    # f1/J = (deltaJ/J) / 2**0.5 + (deltaJ/J)**2 / 8
    # f2/J = (deltaJ/J) / 2**0.5 - (deltaJ/J)**2 / 8
    # f0/J = (deltaJ/J) * (2**0.5)
    
    ratio = dJ_over_J
    
    f1_over_J = ratio / 2**0.5 + ratio**2 / 8
    f2_over_J = ratio / 2**0.5 - ratio**2 / 8
    f0_over_J = ratio * 2**0.5
    
    return -np.stack((f2_over_J, f1_over_J, f0_over_J), axis=1).reshape(-1, 3)


def get_eigenvalue_dimensionless(jintra: np.ndarray, deltaJ: np.ndarray) -> np.ndarray:
    """
    Get dimensionless eigenvalue frequencies (Z/J) from physical parameters.
    
    Args:
        jintra: Array of intramolecular coupling constants
        deltaJ: Array of intermolecular coupling differences
        
    Returns:
        Array of shape (B, 3) with dimensionless eigenvalue frequencies (Z/J)
    """
    # Get physical frequencies
    physical_freqs = get_frequences_ordered(jintra, deltaJ)
    
    # Convert to dimensionless by dividing by |jintra|
    # Use absolute value to handle negative jintra properly
    dimensionless_freqs = physical_freqs / np.abs(jintra.reshape(-1, 1))
    
    return dimensionless_freqs


def make_dimensionless_dataset_from_function(
    num_samples: int,
    func,
    min_Jintra: float = -32,
    max_Jintra: float = -5,
    min_deltaJ: float = -15,
    max_deltaJ: float = -0.1,
    ratio_threshold: float = 100,
    input_mode: str = "1d",  # "1d" or "2d"
) -> Dict[str, torch.Tensor]:
    """
    Generate a dimensionless dataset using a user-supplied function.

    Args:
        num_samples: Number of samples to generate
        func: Function that takes (Jintra, deltaJ) and returns dimensionless labels (Z/J)
        min_Jintra, max_Jintra: Range for Jintra parameter
        min_deltaJ, max_deltaJ: Range for deltaJ parameter
        ratio_threshold: Maximum allowed ratio value (for filtering)
        input_mode: "1d" (dJ/J only) or "2d" (dJ/J and J/dJ)

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
        ratiosX = deltaj / jintra  # dJ/J
        ratiosY = jintra / deltaj  # J/dJ
        # Keep only samples where absolute ratio is within threshold
        valid_mask = np.abs(ratiosX) <= ratio_threshold
        valid_mask &= np.abs(ratiosY) <= ratio_threshold
        jintra = jintra[valid_mask]
        deltaj = deltaj[valid_mask]
        
        # If no valid samples, continue to next iteration
        if len(jintra) == 0:
            continue
        
        # Apply function to get dimensionless labels
        if func == get_perturbation_dimensionless:
            # For perturbation, only need the ratio
            labels = func(deltaj / jintra)
        else:
            # For eigenvalue function, pass both parameters
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
    
    # Calculate dimensionless ratios
    dJ_over_J = deltaj / jintra  # dJ/J
    J_over_dJ = jintra / deltaj  # J/dJ

    # Prepare input features based on input_mode
    if input_mode == "1d":
        # x0 = dJ/J (dimensionless ratio)
        input_features = dJ_over_J.reshape(-1, 1)
    elif input_mode == "2d":
        # x0 = dJ/J, x1 = J/dJ (both dimensionless)
        input_features = np.stack([dJ_over_J, J_over_dJ], axis=1)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    # Store original J values for later scaling back to physical units
    return {
        'train_input': torch.tensor(input_features).float(),
        'train_label': torch.tensor(labels).float(),
        'jintra_values': torch.tensor(jintra).float(),  # Store for scaling back
        'deltaj_values': torch.tensor(deltaj).float(),  # Store for reference
    }


def train_dimensionless_kan_model(
    model: KharKAN,
    data: Dict[str, torch.Tensor],
    config: DimensionlessExperimentConfig,
    lr: float,
    l05_penalty: float,
    device: torch.device
) -> Tuple[List[float], List[float], List[float], float]:
    """
    Train a single dimensionless KAN model.
    
    Returns:
        Tuple of (loss_history, mse_history, rel_err_history, final_mse)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction="none")
    
    inputs = data['train_input'].to(device)
    labels = data['train_label'].to(device)
    
    loss_history = []
    mse_history = []
    rel_err_history = []
    
    pbar = tqdm.tqdm(range(config.epochs), desc=f"Training Dimensionless LR={lr}, L05={l05_penalty}")
    
    for epoch in pbar:
        optimizer.zero_grad()
        preds = model(inputs)
        mse = criterion(preds, labels)
        mse_norm = mse / (torch.abs(labels) + 1e-8)
        l05_loss = model.L05_loss()
        
        # Adaptive loss weighting
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


def extract_and_convert_to_physical_formulas(
    model: KharKAN, input_dim: int = 1, round_digits: int = 2
) -> Dict[str, Optional[sp.Expr]]:
    """
    Extract symbolic formulas from trained dimensionless model and convert to physical units.

    Process:
    1. Extract raw formulas from model (in dimensionless form Z/J)
    2. Replace variable names: x0 -> dJ/J, x1 -> J/dJ
    3. Multiply by J to convert from dimensionless (Z/J) to physical units (Z)
    4. Apply mathematical simplifications (dJ/J * J/dJ = 1)
    5. Simplify and clean the formulas

    Args:
        model: Trained KharKAN model
        input_dim: Number of input dimensions (1 or 2)
        round_digits: Number of digits to round coefficients

    Returns:
        Dictionary with output names as keys and physical unit formulas as values
    """
    raw_formulas = model.symbolic_formula(round_digits=5)
    cleaned_formulas = {}

    x0, x1 = sp.symbols("x_0 x_1")
    # Use proper mathematical symbols for correct simplification
    dJ, J = sp.symbols("dJ J")

    for output_name in ["z_0", "z_1", "z_2"]:
        raw = raw_formulas.get(output_name, None)
        if raw is not None and isinstance(raw, sp.Expr):
            # Step 1: Replace with actual mathematical expressions
            if input_dim == 1:
                # x0 = dJ/J
                e = raw.subs({x0: dJ / J})
            elif input_dim == 2:
                # x0 = dJ/J, x1 = J/dJ
                e = raw.subs({x0: dJ / J, x1: J / dJ})
            else:
                e = raw

            # Step 2: Multiply by J to convert from dimensionless (Z/J) to physical units (Z)
            e = e * J

            # Step 3: Apply mathematical simplifications
            # This will correctly handle cases like (dJ/J) * (J/dJ) = 1
            e = sp.simplify(e)
            
            # Step 4: Clean and round
            e = _fix_powers(e)
            e = _round_expr(e, 5)
            e = _clean_expr(e, eps=1e-5)
            e = _round_expr(e, round_digits)
            
            cleaned_formulas[output_name] = e
        else:
            cleaned_formulas[output_name] = None

    return cleaned_formulas


def scale_predictions_to_physical_units(
    dimensionless_preds: torch.Tensor,
    jintra_values: torch.Tensor
) -> torch.Tensor:
    """
    Scale dimensionless predictions (Z/J) back to physical units (Z).
    
    Args:
        dimensionless_preds: Predictions in dimensionless form (Z/J)
        jintra_values: Original Jintra values for scaling
        
    Returns:
        Predictions in physical units (Z)
    """
    # Multiply by |jintra| to get physical frequencies
    # Use absolute value to handle negative jintra properly
    jintra_abs = torch.abs(jintra_values).unsqueeze(1)  # Shape: (B, 1)
    physical_preds = dimensionless_preds * jintra_abs  # Broadcasting
    
    return physical_preds


def calculate_formula_complexity(formulas: Dict[str, Optional[sp.Expr]]) -> Dict[str, int]:
    """Calculate complexity (number of operations) for each formula."""
    complexities = {}
    for name, formula in formulas.items():
        if formula is not None:
            complexities[name] = sp.count_ops(formula)
        else:
            complexities[name] = float('inf')
    return complexities


def run_single_dimensionless_experiment(
    architecture: Tuple[int, ...],
    data_pert: Dict[str, torch.Tensor],
    data_eigen: Dict[str, torch.Tensor],
    config: DimensionlessExperimentConfig,
    lr: float,
    l05_penalty: float,
    experiment_id: str,
    params_name: str = ""
) -> Dict[str, Any]:
    """Run a single dimensionless experiment configuration."""
    
    results = {
        'experiment_id': experiment_id,
        'architecture': architecture,
        'lr': lr,
        'l05_penalty': l05_penalty,
        'timestamp': time.time()
    }
    
    print(f"\n{'='*60}")
    print(f"Running Dimensionless Experiment: {experiment_id}")
    print(f"Architecture: {architecture}, LR: {lr}, L05: {l05_penalty}")
    print(f"Using parameter set: {params_name}")
    print(f"{'='*60}")
    
    # Run perturbation theory experiment (dimensionless)
    print("\n--- Perturbation Theory Data (Dimensionless) ---")
    model_pert = KharKAN(architecture)

    loss_hist_pert, mse_hist_pert, rel_err_hist_pert, final_mse_pert = train_dimensionless_kan_model(
        model_pert, data_pert, config, lr, l05_penalty, config.device
    )

    # Extract dimensionless formulas and refine them to physical units
    formulas_pert = extract_and_convert_to_physical_formulas(model_pert, architecture[0])
    complexity_pert = calculate_formula_complexity(formulas_pert)

    # For refined formulas, we now have physical units (multiplied by J)
    refined_formulas_pert = formulas_pert  # These are now in physical units
    consistency_flags_pert = {k: True for k in formulas_pert.keys()}
    fix_messages_pert = {k: "Converted from dimensionless (Z/J) to physical units (Z) by multiplying by J" for k in formulas_pert.keys()}
    optimization_results_pert = {k: {"optimization_success": True, "message": "Dimensionless to physical conversion completed"} for k in formulas_pert.keys()}

    # Test the model and scale predictions back to physical units
    with torch.no_grad():
        model_pert.eval()
        dimensionless_preds_pert = model_pert(data_pert["train_input"].to(config.device))
        physical_preds_pert = scale_predictions_to_physical_units(
            dimensionless_preds_pert.cpu(), data_pert["jintra_values"]
        )
        
        # Calculate MSE in physical units for comparison
        # Generate physical labels for comparison
        physical_labels_pert = get_frequences_ordered(
            data_pert["jintra_values"].numpy(), 
            data_pert["deltaj_values"].numpy()
        )
        physical_labels_pert = torch.tensor(physical_labels_pert).float()
        
        physical_mse_pert = torch.nn.functional.mse_loss(physical_preds_pert, physical_labels_pert).item()

    results["perturbation"] = {
        "final_mse": final_mse_pert,
        "physical_mse": physical_mse_pert,
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

    print(f"Perturbation - Final MSE (dimensionless): {final_mse_pert:.2e}")
    print(f"Perturbation - Final MSE (physical units): {physical_mse_pert:.2e}")
    for name, formula in formulas_pert.items():
        if formula is not None:
            print(f"{name}: {formula}")

    # Run eigenvalue experiment (dimensionless)
    print("\n--- Eigenvalue Data (Dimensionless) ---")
    model_eigen = KharKAN(architecture)

    loss_hist_eigen, mse_hist_eigen, rel_err_hist_eigen, final_mse_eigen = (
        train_dimensionless_kan_model(model_eigen, data_eigen, config, lr, l05_penalty, config.device)
    )

    # Extract dimensionless formulas and refine them to physical units
    formulas_eigen = extract_and_convert_to_physical_formulas(model_eigen, architecture[0])
    complexity_eigen = calculate_formula_complexity(formulas_eigen)

    # For refined formulas, we now have physical units (multiplied by J)
    refined_formulas_eigen = formulas_eigen  # These are now in physical units
    consistency_flags_eigen = {k: True for k in formulas_eigen.keys()}
    fix_messages_eigen = {k: "Converted from dimensionless (Z/J) to physical units (Z) by multiplying by J" for k in formulas_eigen.keys()}
    optimization_results_eigen = {k: {"optimization_success": True, "message": "Dimensionless to physical conversion completed"} for k in formulas_eigen.keys()}

    # Test the model and scale predictions back to physical units
    with torch.no_grad():
        model_eigen.eval()
        dimensionless_preds_eigen = model_eigen(data_eigen["train_input"].to(config.device))
        physical_preds_eigen = scale_predictions_to_physical_units(
            dimensionless_preds_eigen.cpu(), data_eigen["jintra_values"]
        )
        
        # Calculate MSE in physical units for comparison
        # Generate physical labels for comparison
        physical_labels_eigen = get_frequences_ordered(
            data_eigen["jintra_values"].numpy(), 
            data_eigen["deltaj_values"].numpy()
        )
        physical_labels_eigen = torch.tensor(physical_labels_eigen).float()
        
        physical_mse_eigen = torch.nn.functional.mse_loss(physical_preds_eigen, physical_labels_eigen).item()

    results["eigenvalue"] = {
        "final_mse": final_mse_eigen,
        "physical_mse": physical_mse_eigen,
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

    print(f"Eigenvalue - Final MSE (dimensionless): {final_mse_eigen:.2e}")
    print(f"Eigenvalue - Final MSE (physical units): {physical_mse_eigen:.2e}")
    for name, formula in formulas_eigen.items():
        if formula is not None:
            print(f"{name}: {formula}")
    
    return results


def save_dimensionless_results_to_json(all_results, output_dir):
    """Save dimensionless experiment results to JSON file."""
    
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
                        "final_mse": data.get("final_mse"),
                        "physical_mse": data.get("physical_mse"),
                        "total_complexity": data.get("total_complexity"),
                        "formulas": {},
                        "refined_formulas": {},
                        "dimensional_consistency": data.get(
                            "dimensional_consistency", {}
                        ),
                        "dimensional_messages": data.get("dimensional_messages", {}),
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
    json_path = output_dir / 'dimensionless_experiment_results.json'
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Dimensionless results saved to: {json_path}")


def main():
    """Main function to run all dimensionless experiments."""
    
    # Setup
    config = DimensionlessExperimentConfig()
    output_dir = Path('experiment_results') / f"dimensionless_experiments_{int(time.time())}"
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
    
    print("\nGenerating dimensionless datasets for all parameter sets...")
    
    for params_name in param_names:
        params = config.data_params[params_name]
        print(f"\n--- Generating dimensionless data for parameter set: {params_name} ---")
        print(f"Parameters: {params}")
        
        # Create datasets with different input dimensions for this parameter set
        datasets_1d = {}
        datasets_2d = {}
        
        for arch in config.architectures:
            input_dim = arch[0]

            if input_dim == 1 and "1d" not in datasets_1d:
                # Generate 1-input datasets - eigenvalue data (dimensionless)
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

                # Generate perturbation data (dimensionless) with same inputs
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

                print(f"1D Dimensionless Dataset shapes for {params_name}:")
                print(f"  Eigenvalue input: {data_eigen_1d['train_input'].shape}")
                print(f"  Eigenvalue label: {data_eigen_1d['train_label'].shape}")
                print(f"  Perturbation input: {data_pert_1d['train_input'].shape}")
                print(f"  Perturbation label: {data_pert_1d['train_label'].shape}")

                print(f"1D Dimensionless Data value ranges for {params_name}:")
                print(
                    f"  Eigenvalue label range: [{data_eigen_1d['train_label'].min():.2e}, {data_eigen_1d['train_label'].max():.2e}]"
                )
                print(
                    f"  Perturbation label range: [{data_pert_1d['train_label'].min():.2e}, {data_pert_1d['train_label'].max():.2e}]"
                )

                datasets_1d = {"pert": data_pert_1d, "eigen": data_eigen_1d}

            elif input_dim == 2 and "2d" not in datasets_2d:
                # Generate 2-input datasets - eigenvalue data (dimensionless)
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

                # Generate perturbation data (dimensionless) with same inputs  
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
                
                print(f"2D Dimensionless Dataset shapes for {params_name}:")
                print(f"  Eigenvalue input: {data_eigen_2d['train_input'].shape}")
                print(f"  Eigenvalue label: {data_eigen_2d['train_label'].shape}")
                print(f"  Perturbation input: {data_pert_2d['train_input'].shape}")
                print(f"  Perturbation label: {data_pert_2d['train_label'].shape}")
                
                datasets_2d = {'pert': data_pert_2d, 'eigen': data_eigen_2d}
        
        # Store datasets for this parameter set
        all_datasets[params_name] = {
            "1d": datasets_1d,
            "2d": datasets_2d,
        }
    
    # Run experiments
    print("\nStarting dimensionless experiments...")
    all_results = []
    experiment_count = 0
    total_experiments = len(param_names) * len(config.architectures) * len(config.learning_rates) * len(config.l05_penalties)
    
    for params_name in param_names:
        print(f"\n{'='*80}")
        print(f"RUNNING DIMENSIONLESS EXPERIMENTS FOR PARAMETER SET: {params_name.upper()}")
        print(f"{'='*80}")

        datasets_1d = all_datasets[params_name]["1d"]
        datasets_2d = all_datasets[params_name]['2d']
        
        for architecture in config.architectures:
            for lr in config.learning_rates:
                for l05_penalty in config.l05_penalties:
                    experiment_count += 1
                    experiment_id = f"dimensionless_{params_name}_arch_{architecture}_lr_{lr}_l05_{l05_penalty}"
                    
                    print(f"\nDimensionless Experiment {experiment_count}/{total_experiments}: {experiment_id}")
                    
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
                        result = run_single_dimensionless_experiment(
                            architecture, data_pert, data_eigen, config, lr, l05_penalty, experiment_id, params_name
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error in dimensionless experiment {experiment_id}: {e}")
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
    save_dimensionless_results_to_json(all_results, output_dir)

    print("\nDimensionless experiments completed!")
    print(f"Results directory: {output_dir}")
    
    # Print summary statistics using utility functions
    print_console_summary(all_results)
    print_detailed_console_summary(all_results)


if __name__ == "__main__":
    main()
