"""
NMRKan - Neural Kolmogorov-Arnold Networks for NMR spectroscopy.

Unified framework for training KAN models on NMR data with support for:
- Single and multi-output training modes
- Multiple dataset types (4CH2, NMR datagen)
- Automatic formula extraction and analysis
- Comprehensive configuration management
"""

# Core models
from .models import KharKAN, DenseKanLayer, _clean_expr

# Training framework
from .training import BaseTrainer, SingleOutputTrainer, MultiOutputTrainer

# Data management
from .data import DatasetFactory, create_dataloader

# Configuration
from .config import TrainingConfig, ModelConfig, DataConfig

# Analysis and utilities
from .analysis import FormulaExtractor
from .utils import (
    calculate_mse,
    calculate_relative_error,
    create_training_plots,
    create_surface_plots,
    save_checkpoint,
    load_checkpoint,
)

# Legacy modules (for backward compatibility)
from .dimensional_analysis import (
    Dimension,
    analyze_nmr_formula,
    check_dimensional_consistency,
    apply_dimensional_analysis,
    get_variable_dimensions,
    get_nmr_output_dimension,
    optimize_formula_coefficients,
)

__all__ = [
    # Core models
    "KharKAN",
    "DenseKanLayer",
    "_clean_expr",
    # Training framework
    "BaseTrainer",
    "SingleOutputTrainer",
    "MultiOutputTrainer",
    # Data management
    "DatasetFactory",
    "create_dataloader",
    # Configuration
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    # Analysis
    "FormulaExtractor",
    # Utilities
    "calculate_mse",
    "calculate_relative_error",
    "create_training_plots",
    "create_surface_plots",
    "save_checkpoint",
    "load_checkpoint",
    # Legacy (backward compatibility)
    "Dimension",
    "analyze_nmr_formula",
    "check_dimensional_consistency",
    "apply_dimensional_analysis",
    "get_variable_dimensions",
    "get_nmr_output_dimension",
    "optimize_formula_coefficients",
]
