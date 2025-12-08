"""Formula extraction and analysis for trained NMR KAN models."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import sympy as sp
import torch
import torch.nn as nn

from ..models import KharKAN

warnings.filterwarnings('ignore', category=UserWarning)


class FormulaExtractor:
    """Extract and analyze symbolic formulas from trained KAN models."""
    
    def __init__(
        self,
        models: Union[KharKAN, List[KharKAN]],
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ):
        """Initialize formula extractor.
        
        Args:
            models: Single model or list of models to analyze
            input_names: Names for input variables
            output_names: Names for output variables
        """
        if isinstance(models, KharKAN):
            self.models = [models]
            self.is_single_model = True
        else:
            self.models = models
            self.is_single_model = False
        
        self.num_models = len(self.models)
        
        # Set default variable names
        if input_names is None:
            # Determine input dimension from first model
            input_dim = self.models[0].layers[0].input_dim
            if input_dim == 1:
                self.input_names = ['deltaJ_ratio']
            elif input_dim == 2:
                self.input_names = ['Jintra', 'deltaJ']
            elif input_dim == 3:
                self.input_names = ['Jintra', 'deltaJ', 'ratio']
            else:
                self.input_names = [f'x_{i}' for i in range(input_dim)]
        else:
            self.input_names = input_names
        
        if output_names is None:
            if self.is_single_model:
                # Check output dimension of the single model
                output_dim = self.models[0].layers[-1].output_dim
                if output_dim == 3:
                    self.output_names = ['f0', 'f1', 'f2']
                elif output_dim == 8:
                    self.output_names = [f'eigenval_{i}' for i in range(8)]
                else:
                    self.output_names = [f'output_{i}' for i in range(output_dim)]
            else:
                # Multi-model case - one output per model
                self.output_names = [f'f{i}' for i in range(self.num_models)]
        else:
            self.output_names = output_names
    
    def extract_formulas(
        self,
        clean: bool = True,
        round_digits: int = 5,
        simplify: bool = True
    ) -> Dict[str, Dict[str, sp.Expr]]:
        """Extract symbolic formulas from all models.
        
        Args:
            clean: Whether to clean small coefficients
            round_digits: Number of digits to round coefficients
            simplify: Whether to simplify expressions
            
        Returns:
            Dictionary mapping model index to formula dictionary
        """
        all_formulas = {}
        
        for i, model in enumerate(self.models):
            model_key = f'model_{i}' if not self.is_single_model else 'single_model'
            
            # Extract formulas from model
            raw_formulas = model.symbolic_formula(clean=clean, round_digits=round_digits)
            
            # Map to meaningful output names
            named_formulas = {}
            for j, (formula_key, expr) in enumerate(raw_formulas.items()):
                if self.is_single_model:
                    # Use output names directly for single model
                    if j < len(self.output_names):
                        output_name = self.output_names[j]
                    else:
                        output_name = f'output_{j}'
                else:
                    # Use model-specific output name for multi-model
                    if i < len(self.output_names):
                        output_name = self.output_names[i]
                    else:
                        output_name = f'output_{i}'
                
                # Replace variable names
                expr_with_names = self._replace_variable_names(expr)
                
                # Simplify if requested
                if simplify:
                    expr_with_names = sp.simplify(expr_with_names)
                
                named_formulas[output_name] = expr_with_names
            
            all_formulas[model_key] = named_formulas
        
        return all_formulas
    
    def _replace_variable_names(self, expr: sp.Expr) -> sp.Expr:
        """Replace generic variable names with meaningful names."""
        # Create substitution mapping
        substitutions = {}
        for i, name in enumerate(self.input_names):
            generic_var = sp.Symbol(f'x_{i}')
            named_var = sp.Symbol(name)
            substitutions[generic_var] = named_var
        
        return expr.xreplace(substitutions)
    
    def calculate_complexity(
        self, 
        formulas: Dict[str, Dict[str, sp.Expr]]
    ) -> Dict[str, Dict[str, Dict[str, Union[int, float]]]]:
        """Calculate complexity metrics for extracted formulas.
        
        Args:
            formulas: Dictionary of formulas from extract_formulas()
            
        Returns:
            Dictionary with complexity metrics for each formula
        """
        complexity_metrics = {}
        
        for model_key, model_formulas in formulas.items():
            model_metrics = {}
            
            for output_name, expr in model_formulas.items():
                metrics = {
                    'num_terms': self._count_terms(expr),
                    'num_operations': self._count_operations(expr),
                    'max_degree': self._get_max_degree(expr),
                    'num_variables': len(expr.free_symbols),
                    'expression_size': len(str(expr)),
                }
                
                model_metrics[output_name] = metrics
            
            complexity_metrics[model_key] = model_metrics
        
        return complexity_metrics
    
    def _count_terms(self, expr: sp.Expr) -> int:
        """Count number of additive terms in expression."""
        if isinstance(expr, sp.Add):
            return len(expr.args)
        else:
            return 1
    
    def _count_operations(self, expr: sp.Expr) -> int:
        """Count total number of operations in expression."""
        if expr.is_Atom:
            return 0
        else:
            return 1 + sum(self._count_operations(arg) for arg in expr.args)
    
    def _get_max_degree(self, expr: sp.Expr) -> int:
        """Get maximum polynomial degree in expression."""
        try:
            degrees = []
            for var in expr.free_symbols:
                degree = sp.degree(expr, var)
                degrees.append(degree)
            return max(degrees) if degrees else 0
        except Exception:
            return 0
    
    def export_formulas(
        self,
        formulas: Dict[str, Dict[str, sp.Expr]],
        export_path: Union[str, Path],
        format: str = 'latex'
    ) -> None:
        """Export formulas to file in specified format.
        
        Args:
            formulas: Dictionary of formulas to export
            export_path: Path to save the exported formulas
            format: Export format ('latex', 'python', 'mathematica', 'text')
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            f.write(f"# Extracted Formulas - {format.upper()} Format\n\n")
            
            for model_key, model_formulas in formulas.items():
                f.write(f"## {model_key}\n\n")
                
                for output_name, expr in model_formulas.items():
                    f.write(f"### {output_name}\n\n")
                    
                    if format == 'latex':
                        formula_str = sp.latex(expr)
                        f.write(f"$$\n{output_name} = {formula_str}\n$$\n\n")
                    elif format == 'python':
                        formula_str = sp.python(expr)
                        f.write(f"# {output_name}\n{formula_str}\n\n")
                    elif format == 'mathematica':
                        formula_str = sp.mathematica_code(expr)
                        f.write(f"(* {output_name} *)\n{formula_str}\n\n")
                    else:  # text
                        f.write(f"{output_name} = {expr}\n\n")
        
        print(f"Formulas exported to: {export_path}")
    
    def compare_with_analytical(
        self,
        analytical_formulas: Dict[str, sp.Expr],
        extracted_formulas: Dict[str, Dict[str, sp.Expr]],
        test_points: Optional[np.ndarray] = None,
        tolerance: float = 1e-6
    ) -> Dict[str, Dict[str, Dict[str, Union[bool, float]]]]:
        """Compare extracted formulas with known analytical expressions.
        
        Args:
            analytical_formulas: Dictionary of known analytical formulas
            extracted_formulas: Dictionary of extracted formulas
            test_points: Array of test points for numerical comparison
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Dictionary with comparison results
        """
        if test_points is None:
            # Generate default test points
            if len(self.input_names) == 1:
                test_points = np.linspace(0.1, 2.0, 50).reshape(-1, 1)
            elif len(self.input_names) == 2:
                x1 = np.linspace(0.1, 10.0, 20)
                x2 = np.linspace(0.01, 1.0, 20)
                X1, X2 = np.meshgrid(x1, x2)
                test_points = np.column_stack([X1.ravel(), X2.ravel()])
            else:
                raise ValueError("Default test points only supported for 1D and 2D inputs")
        
        comparison_results = {}
        
        for model_key, model_formulas in extracted_formulas.items():
            model_results = {}
            
            for output_name, extracted_expr in model_formulas.items():
                if output_name in analytical_formulas:
                    analytical_expr = analytical_formulas[output_name]
                    
                    # Symbolic comparison
                    symbolic_match = sp.simplify(extracted_expr - analytical_expr) == 0
                    
                    # Numerical comparison
                    numerical_errors = []
                    for point in test_points:
                        subs_dict = {sp.Symbol(name): val for name, val in zip(self.input_names, point)}
                        
                        try:
                            analytical_val = float(analytical_expr.subs(subs_dict))
                            extracted_val = float(extracted_expr.subs(subs_dict))
                            error = abs(analytical_val - extracted_val)
                            numerical_errors.append(error)
                        except Exception:
                            numerical_errors.append(float('inf'))
                    
                    max_error = max(numerical_errors) if numerical_errors else float('inf')
                    numerical_match = max_error < tolerance
                    
                    model_results[output_name] = {
                        'symbolic_match': symbolic_match,
                        'numerical_match': numerical_match,
                        'max_numerical_error': max_error,
                        'mean_numerical_error': np.mean(numerical_errors) if numerical_errors else float('inf')
                    }
                else:
                    model_results[output_name] = {
                        'symbolic_match': False,
                        'numerical_match': False,
                        'max_numerical_error': float('inf'),
                        'mean_numerical_error': float('inf'),
                        'note': 'No analytical formula provided for comparison'
                    }
            
            comparison_results[model_key] = model_results
        
        return comparison_results
    
    def generate_report(
        self,
        output_dir: Union[str, Path],
        include_complexity: bool = True,
        include_comparison: bool = False,
        analytical_formulas: Optional[Dict[str, sp.Expr]] = None
    ) -> None:
        """Generate comprehensive formula extraction report.
        
        Args:
            output_dir: Directory to save the report
            include_complexity: Whether to include complexity analysis
            include_comparison: Whether to compare with analytical formulas
            analytical_formulas: Analytical formulas for comparison
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract formulas
        formulas = self.extract_formulas()
        
        # Generate main report
        report_path = output_dir / "formula_extraction_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Formula Extraction Report\n\n")
            f.write(f"**Number of models analyzed:** {self.num_models}\n")
            f.write(f"**Input variables:** {', '.join(self.input_names)}\n")
            f.write(f"**Output variables:** {', '.join(self.output_names)}\n\n")
            
            # Extracted formulas
            f.write("## Extracted Formulas\n\n")
            
            for model_key, model_formulas in formulas.items():
                f.write(f"### {model_key}\n\n")
                
                for output_name, expr in model_formulas.items():
                    f.write(f"**{output_name}:**\n")
                    f.write(f"```\n{expr}\n```\n\n")
            
            # Complexity analysis
            if include_complexity:
                complexity = self.calculate_complexity(formulas)
                f.write("## Complexity Analysis\n\n")
                
                for model_key, model_metrics in complexity.items():
                    f.write(f"### {model_key}\n\n")
                    f.write("| Output | Terms | Operations | Max Degree | Variables | Size |\n")
                    f.write("|--------|-------|------------|------------|-----------|------|\n")
                    
                    for output_name, metrics in model_metrics.items():
                        f.write(f"| {output_name} | {metrics['num_terms']} | "
                               f"{metrics['num_operations']} | {metrics['max_degree']} | "
                               f"{metrics['num_variables']} | {metrics['expression_size']} |\n")
                    f.write("\n")
            
            # Comparison with analytical formulas
            if include_comparison and analytical_formulas:
                comparison = self.compare_with_analytical(analytical_formulas, formulas)
                f.write("## Comparison with Analytical Formulas\n\n")
                
                for model_key, model_results in comparison.items():
                    f.write(f"### {model_key}\n\n")
                    f.write("| Output | Symbolic Match | Numerical Match | Max Error | Mean Error |\n")
                    f.write("|--------|----------------|-----------------|-----------|------------|\n")
                    
                    for output_name, results in model_results.items():
                        f.write(f"| {output_name} | {results['symbolic_match']} | "
                               f"{results['numerical_match']} | {results['max_numerical_error']:.2e} | "
                               f"{results['mean_numerical_error']:.2e} |\n")
                    f.write("\n")
        
        # Export formulas in different formats
        self.export_formulas(formulas, output_dir / "formulas.latex", "latex")
        self.export_formulas(formulas, output_dir / "formulas.py", "python") 
        self.export_formulas(formulas, output_dir / "formulas.txt", "text")
        
        print(f"Formula extraction report generated: {output_dir}")