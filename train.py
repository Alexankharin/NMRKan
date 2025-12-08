#!/usr/bin/env python3
"""
Unified training entry point for NMR KAN models.

This script provides a single interface for training KAN models on NMR data
with support for:
- Single vs multi-output training modes
- 4CH2 eigenvalue vs NMR datagen datasets
- Flexible model configurations
- Automatic formula extraction
- Comprehensive reporting

Usage:
    python train.py --dataset 4ch2 --mode single --config configs/default.yaml
    python train.py --dataset nmr_datagen --mode multi --outputs 3
    python train.py --help
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional, List

import torch
import yaml

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nmrkan.training import SingleOutputTrainer, MultiOutputTrainer
from nmrkan.data import DatasetFactory
from nmrkan.config import TrainingConfig, ModelConfig, DataConfig
from nmrkan.analysis import FormulaExtractor

warnings.filterwarnings('ignore', category=UserWarning)


def load_config_from_yaml(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_configs_from_args(args) -> tuple:
    """Create configuration objects from command line arguments."""
    
    # Load base config if provided
    base_config = {}
    if args.config and Path(args.config).exists():
        base_config = load_config_from_yaml(Path(args.config))
    
    # Training configuration
    training_kwargs = base_config.get('training', {})
    training_kwargs.update({
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'l05_penalty': args.l05_penalty,
        'training_mode': args.mode,
        'optimizer': args.optimizer,
    })
    training_config = TrainingConfig(**training_kwargs)
    
    # Model configuration  
    model_kwargs = base_config.get('model', {})
    if args.architecture:
        # Parse architecture string like "2,8,3,1" 
        layers = tuple(map(int, args.architecture.split(',')))
        model_kwargs['layers'] = layers
    model_config = ModelConfig(**model_kwargs)
    
    # Data configuration
    data_kwargs = base_config.get('data', {})
    data_kwargs.update({
        'dataset_type': args.dataset,
        'num_samples': args.num_samples,
        'num_outputs': args.outputs,
        'dimensionless': args.dimensionless,
        'normalize_inputs': args.normalize_inputs,
        'normalize_outputs': args.normalize_outputs,
    })
    
    # Parameter ranges
    if args.jintra_range:
        jintra_min, jintra_max = map(float, args.jintra_range.split(','))
        data_kwargs.setdefault('parameter_ranges', {})['Jintra'] = (jintra_min, jintra_max)
    
    if args.deltaj_range:
        deltaj_min, deltaj_max = map(float, args.deltaj_range.split(','))
        data_kwargs.setdefault('parameter_ranges', {})['deltaJ'] = (deltaj_min, deltaj_max)
    
    data_config = DataConfig(**data_kwargs)
    
    return training_config, model_config, data_config


def run_training(args) -> None:
    """Run the training process."""
    
    print("=" * 60)
    print("NMR KAN Training - Unified Interface")
    print("=" * 60)
    
    # Create configurations
    training_config, model_config, data_config = create_configs_from_args(args)
    
    print(f"Dataset: {data_config.dataset_type}")
    print(f"Training mode: {training_config.training_mode}")
    print(f"Model architecture: {model_config.layers}")
    print(f"Number of samples: {data_config.num_samples}")
    print(f"Output dimension: {data_config.num_outputs}")
    
    # Set device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset_factory = DatasetFactory(data_config)
    train_loader, val_loader = dataset_factory.create_dataloaders(
        batch_size=training_config.batch_size,
        validation_split=training_config.validation_split,
        shuffle=True
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create experiment directory
    experiment_dir = Path(args.output_dir) if args.output_dir else None
    
    # Choose trainer based on mode
    if training_config.training_mode == 'single':
        print("\nInitializing single-output trainer...")
        trainer = SingleOutputTrainer(
            training_config=training_config,
            model_config=model_config,
            device=device,
            experiment_dir=experiment_dir
        )
        
        # Train the model
        print("\nStarting training...")
        trained_model = trainer.train(train_loader, val_loader)
        
        # Extract formulas
        if args.extract_formulas:
            print("\nExtracting formulas...")
            formula_extractor = FormulaExtractor(
                models=trained_model,
                input_names=get_input_names(data_config),
                output_names=get_output_names(data_config)
            )
            formula_extractor.generate_report(trainer.experiment_dir / "formulas")
        
    elif training_config.training_mode == 'multi':
        print("\nInitializing multi-output trainer...")
        # For multi-output mode, override the model config to have single output per model
        multi_model_config = ModelConfig(
            layers=model_config.layers[:-1] + (1,),  # Change last layer to 1 output
            activations=model_config.activations,
            activation_reprs=model_config.activation_reprs,
            linear_bias=model_config.linear_bias,
            shared_backbone=model_config.shared_backbone
        )
        trainer = MultiOutputTrainer(
            training_config=training_config,
            model_config=multi_model_config,
            device=device,
            experiment_dir=experiment_dir
        )
        # Set the correct number of outputs based on data config
        trainer.num_outputs = data_config.num_outputs
        
        # Train the models
        print("\nStarting training...")
        trained_models = trainer.train(train_loader, val_loader)
        
        # Extract formulas
        if args.extract_formulas:
            print("\nExtracting formulas...")
            formula_extractor = FormulaExtractor(
                models=trained_models,
                input_names=get_input_names(data_config),
                output_names=get_output_names(data_config)
            )
            formula_extractor.generate_report(trainer.experiment_dir / "formulas")
    
    else:
        raise ValueError(f"Invalid training mode: {training_config.training_mode}")
    
    # Generate evaluation report
    if args.evaluate:
        print("\nGenerating evaluation report...")
        generate_evaluation_report(trainer, val_loader)
    
    print(f"\nTraining completed! Results saved to: {trainer.experiment_dir}")


def get_input_names(data_config: DataConfig) -> List[str]:
    """Get appropriate input variable names based on data configuration."""
    if data_config.dimensionless:
        return ['deltaJ_ratio']
    else:
        return ['Jintra', 'deltaJ']


def get_output_names(data_config: DataConfig) -> List[str]:
    """Get appropriate output variable names based on data configuration."""
    if data_config.output_type == 'frequencies':
        if data_config.num_outputs == 3:
            return ['f0', 'f1', 'f2']
        else:
            return [f'f{i}' for i in range(data_config.num_outputs)]
    elif data_config.output_type == 'eigenvalues':
        return [f'eigenval_{i}' for i in range(data_config.num_outputs)]
    else:
        return [f'output_{i}' for i in range(data_config.num_outputs)]


def generate_evaluation_report(trainer, val_loader) -> None:
    """Generate comprehensive evaluation report."""
    
    # Evaluate on validation data
    if hasattr(trainer, 'models'):  # Multi-output trainer
        models = trainer.models
        eval_results = trainer.evaluate(models, val_loader)
        
        # Print results for each model
        for i, results in enumerate(eval_results):
            print(f"\nModel {i} Results:")
            print(f"  MSE: {results['mse']:.6f}")
            print(f"  Relative Error: {results['rel_error']:.6f}")
            print(f"  L0.5 Regularization: {results['l05']:.6f}")
    
    else:  # Single-output trainer  
        model = trainer.model
        eval_results = trainer.evaluate(model, val_loader)
        
        print("\nEvaluation Results:")
        print(f"  Overall MSE: {eval_results['mse']:.6f}")
        print(f"  L0.5 Regularization: {eval_results['l05']:.6f}")
        
        # Per-output results
        for key, value in eval_results.items():
            if key.startswith('mse_output_'):
                output_idx = key.split('_')[-1]
                print(f"  Output {output_idx} MSE: {value:.6f}")
            elif key.startswith('rel_error_output_'):
                output_idx = key.split('_')[-1] 
                print(f"  Output {output_idx} Rel. Error: {value:.6f}")


def create_default_config() -> None:
    """Create default configuration files."""
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    # Default configuration
    default_config = {
        'training': {
            'epochs': 50000,
            'learning_rate': 1e-4,
            'batch_size': 1000,
            'l05_penalty': 0.1,
            'optimizer': 'adam',
            'early_stopping_threshold': 1e-4,
            'early_stopping_patience': 1000,
        },
        'model': {
            'layers': [2, 8, 3, 1],
            'linear_bias': True,
        },
        'data': {
            'dataset_type': '4ch2',
            'num_samples': 10000,
            'num_outputs': 3,
            'parameter_ranges': {
                'Jintra': [0.1, 10.0],
                'deltaJ': [0.01, 1.0]
            },
            'normalize_inputs': True,
            'normalize_outputs': False,
            'dimensionless': False,
        }
    }
    
    config_path = config_dir / "default.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Default configuration created: {config_path}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Unified NMR KAN Training Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model with 4CH2 dataset
  python train.py --dataset 4ch2 --mode single --outputs 3
  
  # Multi-model with NMR datagen
  python train.py --dataset nmr_datagen --mode multi --outputs 3
  
  # Dimensionless training with custom architecture
  python train.py --dataset 4ch2 --mode single --dimensionless --architecture 1,8,8
  
  # Training with configuration file
  python train.py --config configs/default.yaml
  
  # Create default configuration
  python train.py --create-config
        """
    )
    
    # Special commands
    parser.add_argument('--create-config', action='store_true',
                       help='Create default configuration file')
    
    # Dataset configuration
    parser.add_argument('--dataset', choices=['4ch2', 'nmr_datagen'], 
                       default='4ch2', help='Dataset type to use')
    parser.add_argument('--num-samples', type=int, default=10000,
                       help='Number of samples to generate')
    parser.add_argument('--dimensionless', action='store_true',
                       help='Use dimensionless (ratio-based) inputs')
    
    # Model configuration
    parser.add_argument('--mode', choices=['single', 'multi'], default='single',
                       help='Training mode: single model or multiple models')
    parser.add_argument('--architecture', type=str,
                       help='Model architecture as comma-separated integers (e.g., "2,8,3,1")')
    parser.add_argument('--outputs', type=int, default=3,
                       help='Number of output dimensions')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50000,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size')
    parser.add_argument('--l05-penalty', type=float, default=0.1,
                       help='L0.5 regularization penalty')
    parser.add_argument('--optimizer', choices=['adam', 'sgd'], default='adam',
                       help='Optimizer to use')
    
    # Data parameters
    parser.add_argument('--jintra-range', type=str, default=None,
                       help='Jintra parameter range as "min,max"')
    parser.add_argument('--deltaj-range', type=str, default=None,
                       help='deltaJ parameter range as "min,max"')
    parser.add_argument('--normalize-inputs', action='store_true',
                       help='Normalize input features')
    parser.add_argument('--normalize-outputs', action='store_true',
                       help='Normalize output targets')
    
    # Output and analysis
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--extract-formulas', action='store_true', default=True,
                       help='Extract symbolic formulas from trained models')
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Generate evaluation report')
    
    # System configuration
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to use for training')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_default_config()
        return
    
    # Run training
    try:
        run_training(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()