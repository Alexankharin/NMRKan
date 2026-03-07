"""Single output trainer for NMR KAN models."""

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from ..models import KharKAN
from ..config import TrainingConfig, ModelConfig
from ..utils import calculate_mse

warnings.filterwarnings('ignore', category=UserWarning)


class SingleOutputTrainer(BaseTrainer):
    """Trainer for single output NMR KAN models.
    
    This trainer trains a single model to predict all outputs simultaneously,
    using a shared feature extraction backbone with multi-dimensional output.
    """
    
    def train_epoch(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train model for one epoch.
        
        Args:
            model: The KAN model to train
            train_loader: Training data loader
            optimizer: Optimizer for training
            
        Returns:
            Dictionary with loss components for this epoch
        """
        model.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_l05 = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss, loss_components = self.compute_loss(model, predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss_components['total']
            total_mse += loss_components['mse']
            total_mae += loss_components['mae']
            total_l05 += loss_components['l05']
            num_batches += 1

        return {
            'total': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'l05': total_l05 / num_batches
        }
    
    def evaluate(
        self, 
        model: nn.Module, 
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model performance on validation data.
        
        Args:
            model: The model to evaluate
            data_loader: Validation data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_l05 = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_inputs, batch_targets in data_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                predictions = model(batch_inputs)
                loss, loss_components = self.compute_loss(model, predictions, batch_targets)

                total_loss += loss_components['total']
                total_mse += loss_components['mse']
                total_mae += loss_components['mae']
                total_l05 += loss_components['l05']
                num_batches += 1

                all_predictions.append(predictions.cpu())
                all_targets.append(batch_targets.cpu())
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate per-output metrics
        output_metrics = {}
        for i in range(all_predictions.shape[1]):
            pred_i = all_predictions[:, i]
            target_i = all_targets[:, i]
            
            mse_i = nn.MSELoss()(pred_i, target_i).item()
            
            # Relative error
            rel_error_i = torch.mean(torch.abs(pred_i - target_i) / (torch.abs(target_i) + 1e-8)).item()
            
            output_metrics[f'mse_output_{i}'] = mse_i
            output_metrics[f'rel_error_output_{i}'] = rel_error_i
        
        # Overall metrics
        evaluation_results = {
            'total_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'l05': total_l05 / num_batches,
            **output_metrics
        }
        
        return evaluation_results
    
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> KharKAN:
        """Main training loop for single output model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Trained KAN model
        """
        print("Starting single output training...")
        print(f"Model architecture: {self.model_config.layers}")
        print(f"Training on device: {self.device}")
        
        # Create model and optimizer
        self.model = self.create_model()
        self.optimizer = self.create_optimizer(self.model)
        
        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        sparsemax_epoch = (
            int(self.training_config.sparsemax_at * self.training_config.epochs)
            if self.training_config.sparsemax_at > 0 else 0
        )

        progress_bar = tqdm(range(self.training_config.epochs), desc="Training")

        for epoch in progress_bar:
            # Switch to sparsemax at the configured epoch
            if sparsemax_epoch and epoch == sparsemax_epoch:
                self.model.switch_to_sparsemax()
                print(f"\nSwitched to sparsemax at epoch {epoch}/{self.training_config.epochs}")
            # Train one epoch
            train_losses = self.train_epoch(self.model, train_loader, self.optimizer)
            
            # Validation
            if val_loader is not None:
                val_losses = self.evaluate(self.model, val_loader)
                current_val_loss = val_losses['mse']
                
                # Check for improvement
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    self.save_model_checkpoint(
                        self.model, epoch, current_val_loss, is_best=True
                    )
                else:
                    epochs_without_improvement += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Train MSE': f"{train_losses['mse']:.6f}",
                    'Train MAE': f"{train_losses['mae']:.6f}",
                    'Val MSE': f"{current_val_loss:.6f}",
                    'Val MAE': f"{val_losses['mae']:.6f}",
                    'Best Val': f"{best_val_loss:.6f}"
                })
            else:
                current_loss = train_losses['mse'] 
                progress_bar.set_postfix({
                    'Train MSE': f"{current_loss:.6f}",
                    'L0.5': f"{train_losses['l05']:.6f}"
                })
            
            # Log training progress
            self.log_training_step(epoch + 1, train_losses, self.training_config.log_interval)
            
            # Check early stopping
            if self.should_stop_early():
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Save checkpoints less frequently and only save best models automatically
            # Regular interval checkpoints are handled in save_model_checkpoint method
            pass
        
        # Save final checkpoint
        final_loss = val_losses['mse'] if val_loader and 'val_losses' in locals() else train_losses['mse']
        self.save_model_checkpoint(self.model, epoch + 1, final_loss)
        
        # Create training plots
        self.create_training_plots()
        
        print(f"\nTraining completed! Best validation MSE: {best_val_loss:.6f}")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.model