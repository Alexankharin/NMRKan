"""Utility functions for metrics calculation."""

import torch
import torch.nn as nn
from typing import Union


def calculate_mse(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """Calculate Mean Squared Error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MSE value as float
    """
    mse = nn.MSELoss()(predictions, targets)
    return mse.item()


def calculate_relative_error(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    epsilon: float = 1e-8
) -> float:
    """Calculate mean relative error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        epsilon: Small constant to avoid division by zero
        
    Returns:
        Mean relative error as float
    """
    relative_errors = torch.abs(predictions - targets) / (torch.abs(targets) + epsilon)
    return torch.mean(relative_errors).item()


def calculate_r2_score(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """Calculate R² (coefficient of determination).
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        R² score as float
    """
    # Convert to same device
    predictions = predictions.detach()
    targets = targets.detach()
    
    # Calculate mean of targets
    target_mean = torch.mean(targets)
    
    # Total sum of squares
    ss_tot = torch.sum((targets - target_mean) ** 2)
    
    # Residual sum of squares
    ss_res = torch.sum((targets - predictions) ** 2)
    
    # R² score
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return r2.item()


def calculate_mae(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> float:
    """Calculate Mean Absolute Error.
    
    Args:
        predictions: Model predictions
        targets: Target values
        
    Returns:
        MAE value as float
    """
    mae = nn.L1Loss()(predictions, targets)
    return mae.item()