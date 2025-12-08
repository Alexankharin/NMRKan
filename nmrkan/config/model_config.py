"""Model configuration class."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any

import torch


@dataclass
class ModelConfig:
    """Configuration for KAN model architecture."""
    
    # Model architecture
    layers: Tuple[int, ...] = (2, 8, 3, 1)
    
    # Activation functions (None means use default)
    activations: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
    activation_reprs: Optional[List[str]] = None
    
    # Linear layer settings
    linear_bias: bool = True
    
    # Model type specific settings
    shared_backbone: bool = True  # For multi-output training
    
    def __post_init__(self):
        """Set default activations if not provided."""
        if self.activations is None:
            # Use regular functions instead of lambdas to avoid pickle issues
            def identity(x):
                return x
            
            def quadratic(x):
                return x ** 2
            
            self.activations = [
                identity,              # identity
                quadratic,             # quadratic  
                torch.zeros_like       # zero
            ]
        
        if self.activation_reprs is None:
            self.activation_reprs = ["", "**2", "*0"]
        
        # Validate layers
        if len(self.layers) < 2:
            raise ValueError("Model must have at least input and output layers")
        
        # Validate activations and representations match
        if len(self.activations) != len(self.activation_reprs):
            raise ValueError("Number of activations must match activation representations")
    
    @property 
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.layers[0]
    
    @property
    def output_dim(self) -> int:
        """Get output dimension.""" 
        return self.layers[-1]
    
    @property
    def hidden_dims(self) -> Tuple[int, ...]:
        """Get hidden layer dimensions."""
        return self.layers[1:-1]
    
    def get_single_output_config(self, output_idx: int = 0) -> 'ModelConfig':
        """Get configuration for single output model."""
        # Modify last layer to have single output
        new_layers = self.layers[:-1] + (1,)
        
        return ModelConfig(
            layers=new_layers,
            activations=self.activations,
            activation_reprs=self.activation_reprs,
            linear_bias=self.linear_bias,
            shared_backbone=self.shared_backbone
        )