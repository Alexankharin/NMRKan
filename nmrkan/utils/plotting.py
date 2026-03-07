"""Plotting utilities for training visualization."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)

# Set plot style
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('default')


def create_training_plots(
    training_history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Create training progress plots.
    
    Args:
        training_history: Dictionary with training metrics history
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    epochs = training_history.get('epoch', [])
    total_losses = training_history.get('total_loss', [])
    mse_losses = training_history.get('mse_loss', [])
    mae_losses = training_history.get('mae_loss', [])
    l05_losses = training_history.get('l05_loss', [])

    if not epochs:
        print("No training history available for plotting")
        return

    has_mae = mae_losses and len(mae_losses) == len(epochs)

    # Create subplots
    n_rows = 2 if not has_mae else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    fig.suptitle('Training Progress', fontsize=16)

    # Total loss
    axes[0, 0].plot(epochs, total_losses, 'b-', linewidth=2)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # MSE loss
    axes[0, 1].plot(epochs, mse_losses, 'r-', linewidth=2)
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    # L0.5 regularization
    axes[1, 0].plot(epochs, l05_losses, 'g-', linewidth=2)
    axes[1, 0].set_title('L0.5 Regularization')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('L0.5 Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Combined losses
    axes[1, 1].plot(epochs, mse_losses, 'r-', label='MSE', linewidth=2)
    axes[1, 1].plot(epochs, l05_losses, 'g-', label='L0.5', linewidth=2)
    if has_mae:
        axes[1, 1].plot(epochs, mae_losses, 'm-', label='MAE', linewidth=2)
    axes[1, 1].set_title('Combined Losses')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    # MAE loss (dedicated subplot)
    if has_mae:
        axes[2, 0].plot(epochs, mae_losses, 'm-', linewidth=2)
        axes[2, 0].set_title('MAE Loss')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('MAE')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_yscale('log')

        axes[2, 1].plot(epochs, mse_losses, 'r-', label='MSE', linewidth=2)
        axes[2, 1].plot(epochs, mae_losses, 'm-', label='MAE', linewidth=2)
        axes[2, 1].set_title('MSE vs MAE')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_surface_plots(
    X: np.ndarray,
    Y: np.ndarray, 
    Z: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    title: str = "Surface Plot",
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Create 3D surface plots for model predictions.
    
    Args:
        X: X-axis grid values
        Y: Y-axis grid values 
        Z: True Z values
        predictions: Optional model predictions
        title: Plot title
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    fig = plt.figure(figsize=(15, 5))
    
    if predictions is not None:
        # Three subplots: true, predicted, difference
        
        # True surface
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_title('True Surface')
        ax1.set_xlabel('Jintra')
        ax1.set_ylabel('deltaJ')
        ax1.set_zlabel('Output')
        
        # Predicted surface
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, predictions, cmap='plasma', alpha=0.8)
        ax2.set_title('Predicted Surface')
        ax2.set_xlabel('Jintra')
        ax2.set_ylabel('deltaJ')
        ax2.set_zlabel('Output')
        
        # Difference
        ax3 = fig.add_subplot(133, projection='3d')
        diff = Z - predictions
        surf3 = ax3.plot_surface(X, Y, diff, cmap='RdBu', alpha=0.8)
        ax3.set_title('Difference (True - Predicted)')
        ax3.set_xlabel('Jintra')
        ax3.set_ylabel('deltaJ')
        ax3.set_zlabel('Difference')
        
        # Add colorbars
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)
        
    else:
        # Single surface plot
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel('Jintra')
        ax.set_ylabel('deltaJ')
        ax.set_zlabel('Output')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Surface plots saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_comparison_plots(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = False
) -> None:
    """Create prediction vs target comparison plots.
    
    Args:
        targets: True target values
        predictions: Model predictions
        output_names: Names for each output dimension
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """
    num_outputs = targets.shape[1] if len(targets.shape) > 1 else 1
    
    if output_names is None:
        output_names = [f'Output {i}' for i in range(num_outputs)]
    
    # Determine subplot layout
    if num_outputs <= 3:
        cols = num_outputs
        rows = 1
    else:
        cols = 3
        rows = (num_outputs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i in range(num_outputs):
        ax = axes[i] if num_outputs > 1 else axes[0]
        
        if len(targets.shape) > 1:
            y_true = targets[:, i]
            y_pred = predictions[:, i]
        else:
            y_true = targets
            y_pred = predictions
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{output_names[i]} - Predictions vs Targets')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate R²
        from .metrics import calculate_r2_score
        import torch
        r2 = calculate_r2_score(torch.tensor(y_pred), torch.tensor(y_true))
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_outputs, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Prediction vs Target Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()