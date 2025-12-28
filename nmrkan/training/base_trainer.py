"""Base trainer class for NMR KAN models."""

import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import KharKAN
from ..config import TrainingConfig, ModelConfig
from ..utils import calculate_mse, save_checkpoint, create_training_plots

warnings.filterwarnings('ignore', category=UserWarning)


class BaseTrainer(ABC):
    """Base class for NMR KAN model training with common functionality."""
    
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        device: Optional[str] = None,
        experiment_dir: Optional[Path] = None,
    ):
        """Initialize base trainer.
        
        Args:
            training_config: Training configuration parameters
            model_config: Model architecture configuration  
            device: Device to use for training ('cuda', 'cpu', or None for auto)
            experiment_dir: Directory to save experiment results
        """
        self.training_config = training_config
        self.model_config = model_config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set up experiment directory
        if experiment_dir is None:
            timestamp = int(time.time())
            experiment_dir = Path(f"experiment_results/unified_experiments_{timestamp}")
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.model: Optional[KharKAN] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_history: Dict[str, List[float]] = {
            'epoch': [],
            'total_loss': [],
            'mse_loss': [],
            'l05_loss': []
        }
        
    def create_model(self) -> KharKAN:
        """Create model instance based on model configuration."""
        model = KharKAN(
            layers=self.model_config.layers,
            activations=self.model_config.activations,
            activation_reprs=self.model_config.activation_reprs,
            linear_bias=self.model_config.linear_bias,
            include_abs=self.model_config.include_abs,
        )
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for the model."""
        if self.training_config.optimizer == 'adam':
            return torch.optim.Adam(
                model.parameters(), 
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == 'sgd':
            return torch.optim.SGD(
                model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=self.training_config.momentum,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer}")
    
    def compute_loss(
        self, 
        model: nn.Module, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute training loss with regularization.
        
        Args:
            model: The model being trained
            predictions: Model predictions
            targets: Target values
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # MSE loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # L0.5 regularization for sparsity
        l05_loss = model.L05_loss() if hasattr(model, 'L05_loss') else torch.tensor(0.0)
        
        # Total loss
        total_loss = mse_loss + self.training_config.l05_penalty * l05_loss
        
        loss_components = {
            'mse': mse_loss.item(),
            'l05': l05_loss.item() if isinstance(l05_loss, torch.Tensor) else l05_loss,
            'total': total_loss.item()
        }
        
        return total_loss, loss_components
    
    def should_stop_early(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.training_history['mse_loss']) < self.training_config.early_stopping_patience:
            return False
        
        recent_losses = self.training_history['mse_loss'][-self.training_config.early_stopping_patience:]
        min_loss = min(recent_losses)
        
        return min_loss < self.training_config.early_stopping_threshold
    
    def log_training_step(
        self, 
        epoch: int, 
        loss_components: Dict[str, float],
        log_interval: int = 1000
    ) -> None:
        """Log training progress."""
        self.training_history['epoch'].append(epoch)
        self.training_history['total_loss'].append(loss_components['total'])
        self.training_history['mse_loss'].append(loss_components['mse'])
        self.training_history['l05_loss'].append(loss_components['l05'])
        
        if epoch % log_interval == 0 or epoch == 1:
            print(f"Epoch {epoch}: MSE={loss_components['mse']:.6f}, "
                  f"L0.5={loss_components['l05']:.6f}, "
                  f"Total={loss_components['total']:.6f}")
    
    def save_model_checkpoint(
        self, 
        model: nn.Module, 
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint with intelligent cleanup."""
        # Create checkpoint data without unpickleable objects
        training_config_dict = dict(self.training_config.__dict__)
        model_config_dict = dict(self.model_config.__dict__)
        
        # Remove unpickleable activation functions
        if 'activations' in model_config_dict:
            model_config_dict.pop('activations')
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'loss': loss,
            'training_config': training_config_dict,
            'model_config': model_config_dict,
            'training_history': self.training_history
        }
        
        # Save best model (always keep)
        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            save_checkpoint(checkpoint_data, best_path)
            #print(f"New best model saved (epoch {epoch}, loss {loss:.6f})")
        
        # Save regular checkpoint only at specified intervals
        if epoch % self.training_config.save_interval == 0 or is_best:
            checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(checkpoint_data, checkpoint_path)
            
            # Cleanup old checkpoints (keep only last 3 regular checkpoints)
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Remove old checkpoint files, keeping only the most recent ones."""
        from ..utils.checkpoints import cleanup_old_checkpoints
        cleanup_old_checkpoints(self.experiment_dir, keep_last_n, "checkpoint_epoch_*.pt")
    
    def create_training_plots(self) -> None:
        """Create and save training progress plots."""
        if len(self.training_history['epoch']) > 0:
            plot_path = self.experiment_dir / "training_curves.png"
            create_training_plots(self.training_history, save_path=plot_path)
    
    @abstractmethod
    def train_epoch(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train for one epoch. Must be implemented by subclasses."""
        pass
    
    @abstractmethod 
    def evaluate(
        self, 
        model: nn.Module, 
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model performance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> KharKAN:
        """Main training loop. Must be implemented by subclasses."""
        pass