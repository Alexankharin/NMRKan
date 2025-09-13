"""Dataset generation functions for NMR symbolic regression experiments."""

import numpy as np
import torch
from typing import Dict, Callable

from .nmr_datagen import CalcFreqsIdealized

__all__ = [
    "make_dataset_from_function",
    "make_dimensionless_dataset_from_function", 
    "get_eigenvalue_dimensionless",
    "get_perturbation_dimensionless",
    "get_perturbation",
]


def get_perturbation(Jintra: np.ndarray, deltaJ: np.ndarray) -> np.ndarray:
    """Calculate perturbation theory frequencies for given Jintra and deltaJ."""
    # f2 = (deltaJ/sqrt(2)) + (deltaJ^2/8J)
    # f1 = (deltaJ/sqrt(2)) - (deltaJ^2/8J)  
    # f0 = deltaJ*sqrt(2)
    f2 = deltaJ / np.sqrt(2) + deltaJ**2 / (8 * Jintra)
    f1 = deltaJ / np.sqrt(2) - deltaJ**2 / (8 * Jintra)
    f0 = deltaJ * np.sqrt(2)
    return -np.stack((f2, f1, f0), axis=1)


def get_perturbation_dimensionless(dJ_over_J: np.ndarray) -> np.ndarray:
    """Calculate dimensionless perturbation theory frequencies.
    
    Args:
        dJ_over_J: Array of deltaJ/Jintra ratios
        
    Returns:
        Array of shape (N, 3) with dimensionless frequencies [f2/J, f1/J, f0/J]
    """
    # Dimensionless perturbation formulas:
    # f2/J = (deltaJ/J) / sqrt(2) + (deltaJ/J)^2 / 8
    # f1/J = (deltaJ/J) / sqrt(2) - (deltaJ/J)^2 / 8
    # f0/J = (deltaJ/J) * sqrt(2)
    
    ratio = dJ_over_J
    
    f1_over_J = ratio / 2**0.5 + ratio**2 / 8
    f2_over_J = ratio / 2**0.5 - ratio**2 / 8
    f0_over_J = ratio * 2**0.5
    
    return -np.stack((f2_over_J, f1_over_J, f0_over_J), axis=1).reshape(-1, 3)


def get_eigenvalue_dimensionless(jintra: np.ndarray, deltaJ: np.ndarray) -> np.ndarray:
    """Calculate dimensionless eigenvalue frequencies (Z/J) from CalcFreqsIdealized.
    
    Args:
        jintra: Array of Jintra values
        deltaJ: Array of deltaJ values
        
    Returns:
        Array of shape (N, 3) with dimensionless frequencies normalized by |Jintra|
    """
    freqs = []
    for j, d in zip(jintra.flatten(), deltaJ.flatten()):
        freq_hz = CalcFreqsIdealized(j, d)  # Returns physical frequencies
        # Normalize by |Jintra| to get dimensionless Z/J
        freq_dimensionless = np.array(freq_hz) / abs(j)
        freqs.append(freq_dimensionless)
    
    return np.array(freqs, dtype=np.float32)


def make_dataset_from_function(
    num_samples: int,
    func: Callable,
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


def make_dimensionless_dataset_from_function(
    num_samples: int,
    func: Callable,
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