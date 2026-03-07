"""Multi-output trainer for NMR KAN models."""

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


class MultiOutputTrainer(BaseTrainer):
    """Trainer for multi-output NMR KAN models.
    
    This trainer creates separate models for each output, allowing for
    specialized learning of individual target functions.
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        device: Optional[str] = None,
        experiment_dir: Optional[str] = None,
    ):
        """Initialize multi-output trainer."""
        super().__init__(training_config, model_config, device, experiment_dir)
        
        # Multi-output specific attributes
        self.models: List[KharKAN] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        # For multi-output training, we create separate models for each output
        # If the original model has multiple outputs (e.g., last layer = 3), 
        # we create 3 separate models each with 1 output
        if model_config.layers[-1] > 1:
            self.num_outputs = model_config.layers[-1]
        else:
            # If already configured for single output, assume 3 separate models
            self.num_outputs = 3
        
        # Training history for each model
        self.multi_training_history: List[Dict[str, List[float]]] = []
        for i in range(self.num_outputs):
            self.multi_training_history.append({
                'epoch': [],
                'total_loss': [],
                'mse_loss': [],
                'mae_loss': [],
                'l05_loss': []
            })
    
    def create_models(self) -> List[KharKAN]:
        """Create separate models for each output."""
        models = []
        
        # Create single-output configuration
        single_config = self.model_config.get_single_output_config()
        
        for i in range(self.num_outputs):
            model = KharKAN(
                layers=single_config.layers,
                activations=single_config.activations,
                activation_reprs=single_config.activation_reprs,
                linear_bias=single_config.linear_bias,
                include_abs=single_config.include_abs,
            )
            models.append(model.to(self.device))
        
        return models
    
    def create_optimizers(self, models: List[nn.Module]) -> List[torch.optim.Optimizer]:
        """Create optimizers for each model."""
        optimizers = []
        
        for model in models:
            optimizer = self.create_optimizer(model)
            optimizers.append(optimizer)
        
        return optimizers
    
    def train_epoch(
        self, 
        models: List[nn.Module], 
        train_loader: DataLoader, 
        optimizers: List[torch.optim.Optimizer]
    ) -> List[Dict[str, float]]:
        """Train all models for one epoch.
        
        Args:
            models: List of models to train
            train_loader: Training data loader
            optimizers: List of optimizers for each model
            
        Returns:
            List of loss dictionaries for each model
        """
        # Set all models to training mode
        for model in models:
            model.train()
        
        epoch_losses = []
        for i in range(self.num_outputs):
            epoch_losses.append({
                'total': 0.0,
                'mse': 0.0,
                'mae': 0.0,
                'l05': 0.0
            })

        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)

            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                target_i = batch_targets[:, i:i+1]

                optimizer.zero_grad()
                prediction_i = model(batch_inputs)
                loss, loss_components = self.compute_loss(model, prediction_i, target_i)
                loss.backward()
                optimizer.step()

                epoch_losses[i]['total'] += loss_components['total']
                epoch_losses[i]['mse'] += loss_components['mse']
                epoch_losses[i]['mae'] += loss_components['mae']
                epoch_losses[i]['l05'] += loss_components['l05']
            
            num_batches += 1
        
        # Average losses
        for i in range(self.num_outputs):
            for key in epoch_losses[i]:
                epoch_losses[i][key] /= num_batches
        
        return epoch_losses
    
    def evaluate(
        self, 
        models: List[nn.Module], 
        data_loader: DataLoader
    ) -> List[Dict[str, float]]:
        """Evaluate all models on validation data.
        
        Args:
            models: List of models to evaluate
            data_loader: Validation data loader
            
        Returns:
            List of evaluation metrics for each model
        """
        # Set all models to evaluation mode
        for model in models:
            model.eval()
        
        eval_losses = []
        for i in range(self.num_outputs):
            eval_losses.append({
                'total': 0.0,
                'mse': 0.0,
                'mae': 0.0,
                'l05': 0.0
            })

        all_predictions = [[] for _ in range(self.num_outputs)]
        all_targets = [[] for _ in range(self.num_outputs)]

        num_batches = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in data_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                for i, model in enumerate(models):
                    target_i = batch_targets[:, i:i+1]
                    prediction_i = model(batch_inputs)
                    loss, loss_components = self.compute_loss(model, prediction_i, target_i)

                    eval_losses[i]['total'] += loss_components['total']
                    eval_losses[i]['mse'] += loss_components['mse']
                    eval_losses[i]['mae'] += loss_components['mae']
                    eval_losses[i]['l05'] += loss_components['l05']
                    
                    # Store predictions and targets
                    all_predictions[i].append(prediction_i.cpu())
                    all_targets[i].append(target_i.cpu())
                
                num_batches += 1
        
        # Average losses and calculate additional metrics
        evaluation_results = []
        
        for i in range(self.num_outputs):
            # Average losses
            for key in eval_losses[i]:
                eval_losses[i][key] /= num_batches
            
            # Concatenate predictions and targets
            pred_i = torch.cat(all_predictions[i], dim=0)
            target_i = torch.cat(all_targets[i], dim=0)
            
            # Calculate relative error
            rel_error = torch.mean(torch.abs(pred_i - target_i) / (torch.abs(target_i) + 1e-8)).item()
            
            result_i = {
                **eval_losses[i],
                'rel_error': rel_error
            }
            evaluation_results.append(result_i)
        
        return evaluation_results
    
    def train(
        self, 
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> List[KharKAN]:
        """Main training loop for multi-output models.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            List of trained KAN models
        """
        print("Starting multi-output training...")
        print(f"Number of output models: {self.num_outputs}")
        print(f"Model architecture: {self.model_config.layers}")
        print(f"Training on device: {self.device}")
        
        # Create models and optimizers
        self.models = self.create_models()
        self.optimizers = self.create_optimizers(self.models)
        
        # Training state tracking
        best_val_losses = [float('inf')] * self.num_outputs
        epochs_without_improvement = [0] * self.num_outputs
        
        progress_bar = tqdm(range(self.training_config.epochs), desc="Training")
        
        for epoch in progress_bar:
            # Train one epoch
            train_losses = self.train_epoch(self.models, train_loader, self.optimizers)
            
            # Validation
            if val_loader is not None:
                val_losses = self.evaluate(self.models, val_loader)
                
                # Track best models
                improved_models = []
                for i in range(self.num_outputs):
                    current_val_loss = val_losses[i]['mse']
                    
                    if current_val_loss < best_val_losses[i]:
                        best_val_losses[i] = current_val_loss
                        epochs_without_improvement[i] = 0
                        improved_models.append(i)
                        
                        # Save best model for this output
                        self.save_model_checkpoint(
                            self.models[i], epoch, current_val_loss, 
                            is_best=True, model_idx=i
                        )
                    else:
                        epochs_without_improvement[i] += 1
                
                # Update progress bar with average metrics
                avg_train_mse = sum(loss['mse'] for loss in train_losses) / len(train_losses)
                avg_train_mae = sum(loss['mae'] for loss in train_losses) / len(train_losses)
                avg_val_mse = sum(loss['mse'] for loss in val_losses) / len(val_losses)
                avg_val_mae = sum(loss['mae'] for loss in val_losses) / len(val_losses)
                avg_best_val = sum(best_val_losses) / len(best_val_losses)

                progress_bar.set_postfix({
                    'Avg Train MSE': f"{avg_train_mse:.6f}",
                    'Avg Train MAE': f"{avg_train_mae:.6f}",
                    'Avg Val MSE': f"{avg_val_mse:.6f}",
                    'Avg Val MAE': f"{avg_val_mae:.6f}",
                    'Avg Best Val': f"{avg_best_val:.6f}",
                    'Improved': len(improved_models)
                })
            else:
                avg_train_mse = sum(loss['mse'] for loss in train_losses) / len(train_losses)
                avg_l05 = sum(loss['l05'] for loss in train_losses) / len(train_losses)
                
                progress_bar.set_postfix({
                    'Avg Train MSE': f"{avg_train_mse:.6f}",
                    'Avg L0.5': f"{avg_l05:.6f}"
                })
            
            # Log training progress for each model
            for i, loss_dict in enumerate(train_losses):
                self.log_training_step_multi(
                    epoch + 1, loss_dict, i, self.training_config.log_interval
                )
            
            # Check early stopping (all models must meet criteria)
            if self.should_stop_early_multi():
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
            
            # Checkpoints are saved automatically in save_model_checkpoint when needed
            # This reduces checkpoint frequency and clutter
            pass
        
        # Save final checkpoints for all models
        for i, model in enumerate(self.models):
            final_loss = val_losses[i]['mse'] if val_loader else train_losses[i]['mse']
            self.save_model_checkpoint(model, epoch + 1, final_loss, model_idx=i)
        
        # Create training plots
        self.create_training_plots_multi()
        
        avg_best_val = sum(best_val_losses) / len(best_val_losses)
        print(f"\nTraining completed! Average best validation MSE: {avg_best_val:.6f}")
        print(f"Results saved to: {self.experiment_dir}")
        
        return self.models
    
    def log_training_step_multi(
        self, 
        epoch: int, 
        loss_components: Dict[str, float],
        model_idx: int,
        log_interval: int = 1000
    ) -> None:
        """Log training progress for a specific model."""
        history = self.multi_training_history[model_idx]
        
        history['epoch'].append(epoch)
        history['total_loss'].append(loss_components['total'])
        history['mse_loss'].append(loss_components['mse'])
        history['mae_loss'].append(loss_components['mae'])
        history['l05_loss'].append(loss_components['l05'])

        if epoch % log_interval == 0 or epoch == 1:
            print(f"Model {model_idx} - Epoch {epoch}: "
                  f"MSE={loss_components['mse']:.6f}, "
                  f"MAE={loss_components['mae']:.6f}, "
                  f"L0.5={loss_components['l05']:.6f}, "
                  f"Total={loss_components['total']:.6f}")
    
    def should_stop_early_multi(self) -> bool:
        """Check if early stopping criteria are met for all models."""
        for i in range(self.num_outputs):
            history = self.multi_training_history[i]
            
            if len(history['mse_loss']) < self.training_config.early_stopping_patience:
                return False
            
            recent_losses = history['mse_loss'][-self.training_config.early_stopping_patience:]
            min_loss = min(recent_losses)
            
            if min_loss >= self.training_config.early_stopping_threshold:
                return False
        
        return True  # All models meet early stopping criteria
    
    def save_model_checkpoint(
        self, 
        model: nn.Module, 
        epoch: int,
        loss: float,
        is_best: bool = False,
        model_idx: int = 0
    ) -> None:
        """Save model checkpoint with model index and cleanup."""
        # Create checkpoint data without unpickleable objects
        training_config_dict = dict(self.training_config.__dict__)
        model_config_dict = dict(self.model_config.__dict__)
        
        # Remove unpickleable activation functions
        if 'activations' in model_config_dict:
            model_config_dict.pop('activations')
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizers[model_idx].state_dict() if self.optimizers else None,
            'loss': loss,
            'model_idx': model_idx,
            'training_config': training_config_dict,
            'model_config': model_config_dict,
            'training_history': self.multi_training_history[model_idx]
        }
        
        # Save best model (always keep)
        if is_best:
            best_path = self.experiment_dir / f"model_{model_idx}_best.pt"
            torch.save(checkpoint_data, best_path)
            print(f"New best model {model_idx} saved (epoch {epoch}, loss {loss:.6f})")
        
        # Save regular checkpoint only at specified intervals or if best
        if epoch % self.training_config.save_interval == 0 or is_best:
            checkpoint_path = self.experiment_dir / f"model_{model_idx}_checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            
            # Cleanup old checkpoints for this model (keep only last 2 per model)
            self._cleanup_model_checkpoints(model_idx)
    
    def _cleanup_model_checkpoints(self, model_idx: int, keep_last_n: int = 2) -> None:
        """Remove old checkpoint files for a specific model."""
        from ..utils.checkpoints import cleanup_old_checkpoints
        pattern = f"model_{model_idx}_checkpoint_epoch_*.pt"
        cleanup_old_checkpoints(self.experiment_dir, keep_last_n, pattern)
    
    def create_training_plots_multi(self) -> None:
        """Create training plots for all models."""
        for i, history in enumerate(self.multi_training_history):
            if len(history['epoch']) > 0:
                plot_path = self.experiment_dir / f"model_{i}_training_curves.png"
                # Use utility function for individual model
                from ..utils import create_training_plots
                create_training_plots(history, save_path=plot_path)