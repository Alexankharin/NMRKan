"""Model configuration class."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from ..activations import default_activation_bank


@dataclass
class ModelConfig:
    """Configuration for KAN model architecture."""
    
    # Model architecture
    layers: Tuple[int, ...] = (2, 8, 3, 1)
    
    # Activation functions (None means use default)
    activations: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
    activation_reprs: Optional[List[str]] = None
    include_abs: bool = True
    
    # Linear layer settings
    linear_bias: bool = True
    
    # Model type specific settings
    shared_backbone: bool = True  # For multi-output training
    
    def __post_init__(self):
        """Set default activations if not provided."""
        default_activations, default_reprs = default_activation_bank(include_abs=self.include_abs)

        if self.activations is None:
            self.activations = default_activations
        
        if self.activation_reprs is None:
            if self.activations is default_activations:
                self.activation_reprs = default_reprs
            else:
                raise ValueError(
                    "activation_reprs must be provided when custom activations are specified"
                )
        
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
            include_abs=self.include_abs,
            linear_bias=self.linear_bias,
            shared_backbone=self.shared_backbone
        )