#!/usr/bin/env python3
"""
NMRKan - Unified command-line interface for NMR symbolic regression experiments.

This script provides a single entry point for all NMR-KAN operations:
- Running experiments (regular and dimensionless)
- Symbolic regression
- Grid search optimization
- Genetic algorithm optimization

Usage:
    python main.py experiment [--mode regular|dimensionless|both] [--config CONFIG]
    python main.py symbolic [--config CONFIG]
    python main.py grid-search [--config CONFIG]
    python main.py genetic [--config CONFIG]
    python main.py --help
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add the current directory to the path
if "." not in sys.path:
    sys.path.append(".")


def run_experiments(mode: str = "both", config: Optional[str] = None) -> int:
    """Run KAN experiments."""
    print(f"Running experiments in {mode} mode...")
    
    cmd = [sys.executable, "experiments/run_experiments.py", "--mode", mode]
    if config:
        cmd.extend(["--config", config])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running experiments: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: experiments/run_experiments.py not found")
        return 1


def run_symbolic_regression(config: Optional[str] = None) -> int:
    """Run symbolic regression."""
    print("Running symbolic regression...")
    
    cmd = [sys.executable, "legacy/run_symbolic_regressor.py"]
    if config:
        cmd.extend(["--config", config])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running symbolic regression: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: legacy/run_symbolic_regressor.py not found")
        return 1


def run_grid_search(config: Optional[str] = None) -> int:
    """Run grid search optimization."""
    print("Running grid search...")
    
    cmd = [sys.executable, "legacy/run_grid_search.py"]
    if config:
        cmd.extend(["--config", config])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running grid search: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: legacy/run_grid_search.py not found")
        return 1


def run_genetic_algorithm(config: Optional[str] = None) -> int:
    """Run genetic algorithm optimization."""
    print("Running genetic algorithm...")
    
    cmd = [sys.executable, "legacy/genetic_KAN.py"]
    if config:
        cmd.extend(["--config", config])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running genetic algorithm: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: legacy/genetic_KAN.py not found")
        return 1


def run_normalized_eigenvalues(config: Optional[str] = None) -> int:
    """Run normalized eigenvalue KAN training."""
    print("Running normalized eigenvalue KAN training...")
    
    cmd = [sys.executable, "experiments/run_experiments.py", "--mode", "normalized-eigenvalues"]
    if config:
        cmd.extend(["--config", config])
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running normalized eigenvalue training: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: experiments/run_experiments.py not found")
        return 1


def list_available_configs() -> None:
    """List available configuration files."""
    config_dir = Path("configs")
    if config_dir.exists():
        config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
        if config_files:
            print("Available configuration files:")
            for config_file in config_files:
                print(f"  {config_file}")
        else:
            print("No configuration files found in configs/ directory")
    else:
        print("No configs/ directory found")


def show_status() -> None:
    """Show repository status and available operations."""
    print("NMRKan Repository Status")
    print("=" * 40)
    
    # Check for required files
    required_files = [
        "experiments/run_experiments.py",
        "legacy/run_symbolic_regressor.py", 
        "legacy/run_grid_search.py",
        "legacy/genetic_KAN.py",
        "data_generation/__init__.py",
        "nmrkan/__init__.py"
    ]
    
    print("\nRequired files:")
    for file_path in required_files:
        exists = Path(file_path).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
    
    # Check for result directories
    print("\nRecent results:")
    results_dir = Path("experiment_results")
    if results_dir.exists():
        recent_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]
        for result_dir in recent_dirs:
            print(f"  {result_dir.name}")
    else:
        print("  No experiment results found")
    
    print("\nAvailable operations:")
    print("  experiment     - Run KAN experiments")
    print("  symbolic       - Run symbolic regression")
    print("  grid-search    - Run grid search optimization")
    print("  genetic        - Run genetic algorithm optimization")
    print("  normalized-eigenvalues - Train KAN with normalized eigenvalue data")
    print("  status         - Show this status information")
    print("  list-configs   - List available configuration files")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="NMRKan - Unified interface for NMR symbolic regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py experiment --mode regular
  python main.py experiment --mode dimensionless
  python main.py experiment --mode both
  python main.py symbolic
  python main.py grid-search
  python main.py genetic
  python main.py normalized-eigenvalues
  python main.py status
  python main.py list-configs
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Experiment subcommand
    exp_parser = subparsers.add_parser("experiment", help="Run KAN experiments")
    exp_parser.add_argument("--mode", choices=["regular", "dimensionless", "both"], 
                           default="both", help="Experiment mode to run")
    exp_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Symbolic regression subcommand
    sym_parser = subparsers.add_parser("symbolic", help="Run symbolic regression")
    sym_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Grid search subcommand
    grid_parser = subparsers.add_parser("grid-search", help="Run grid search optimization")
    grid_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Genetic algorithm subcommand
    genetic_parser = subparsers.add_parser("genetic", help="Run genetic algorithm optimization")
    genetic_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Normalized eigenvalues subcommand
    norm_eigen_parser = subparsers.add_parser("normalized-eigenvalues", help="Train KAN with normalized eigenvalue data")
    norm_eigen_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Status subcommand
    subparsers.add_parser("status", help="Show repository status")
    
    # List configs subcommand
    subparsers.add_parser("list-configs", help="List available configuration files")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate handler
    if args.command == "experiment":
        return run_experiments(args.mode, args.config)
    elif args.command == "symbolic":
        return run_symbolic_regression(args.config)
    elif args.command == "grid-search":
        return run_grid_search(args.config)
    elif args.command == "genetic":
        return run_genetic_algorithm(args.config)
    elif args.command == "normalized-eigenvalues":
        return run_normalized_eigenvalues(args.config)
    elif args.command == "status":
        show_status()
        return 0
    elif args.command == "list-configs":
        list_available_configs()
        return 0
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())