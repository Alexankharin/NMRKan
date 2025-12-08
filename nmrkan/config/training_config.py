"""Training configuration class."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training parameters
    epochs: int = 50000
    learning_rate: float = 1e-4
    batch_size: int = 1000
    
    # Optimizer settings
    optimizer: str = "adam"  # 'adam' or 'sgd'
    weight_decay: float = 0.0
    momentum: float = 0.9  # for SGD
    
    # Loss function parameters
    l05_penalty: float = 0.1
    
    # Early stopping
    early_stopping_threshold: float = 1e-4
    early_stopping_patience: int = 1000
    
    # Checkpointing
    save_interval: int = 10000  # Save less frequently to reduce clutter
    log_interval: int = 1000
    
    # Training mode
    training_mode: str = "single"  # 'single' or 'multi'
    
    # Validation
    validation_split: float = 0.2
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.optimizer not in ['adam', 'sgd']:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
        if self.training_mode not in ['single', 'multi']:
            raise ValueError(f"Unsupported training mode: {self.training_mode}")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Validation split must be between 0 and 1")
        
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")