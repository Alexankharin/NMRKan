import sympy as sp
import numpy as np
import torch
from typing import Dict, Tuple, Optional


class Dimension:
    """Represents a physical dimension as a mapping from base‑unit names to exponents.

    Example
    -------
    >>> hertz = Dimension(Hz=1)       # frequency
    >>> period = Dimension(Hz=-1)     # 1 / frequency
    >>> dimensionless = Dimension()   # no units at all
    """

    _TOL = 1e-12

    def __init__(self, **unit_powers: float):
        # Store only non‑zero exponents (within tolerance)
        self._powers: Dict[str, float] = {
            u: float(p) for u, p in unit_powers.items() if abs(p) > self._TOL
        }

    # ------------------------------------------------------------------
    # Fundamental helpers
    # ------------------------------------------------------------------
    def _clean(self):
        """Remove near‑zero entries (helps after math ops)."""
        self._powers = {u: p for u, p in self._powers.items() if abs(p) > self._TOL}

    def copy(self) -> "Dimension":
        d = Dimension()
        d._powers = dict(self._powers)
        return d

    # ------------------------------------------------------------------
    # Comparisons & display
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        # Compare every unit that appears in either operand
        for unit in set(self._powers) | set(other._powers):
            if abs(self._powers.get(unit, 0.0) - other._powers.get(unit, 0.0)) > self._TOL:
                return False
        return True

    def is_dimensionless(self) -> bool:
        """Return **True** if the dimension has *no* units (all exponents ≈ 0)."""
        return not self._powers  # empty dict → dimensionless

    def __repr__(self) -> str:  # pragma: no cover
        return f"Dimension({self._powers})"

    def __str__(self) -> str:  # pragma: no cover
        if self.is_dimensionless():
            return "dimensionless"
        pieces = []
        for unit, power in sorted(self._powers.items()):
            if abs(power - 1) < self._TOL:
                pieces.append(unit)
            else:
                pieces.append(f"{unit}^{power:g}")
        return "*".join(pieces)

    # ------------------------------------------------------------------
    # Arithmetic ( + permitted only for like‑dimensions )
    # ------------------------------------------------------------------
    def __add__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            raise TypeError("Can only add Dimension objects")
        if self != other:
            raise ValueError(f"Cannot add incompatible dimensions: {self} + {other}")
        return self.copy()  # unchanged (identical dims)

    __radd__ = __add__

    def _combine(self, other: "Dimension", sign: int) -> "Dimension":
        result = self.copy()
        for unit, power in other._powers.items():
            result._powers[unit] = result._powers.get(unit, 0.0) + sign * power
        result._clean()
        return result

    def __mul__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            raise TypeError("Can only multiply Dimension objects")
        return self._combine(other, sign=+1)

    __rmul__ = __mul__

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            raise TypeError("Can only divide Dimension objects")
        return self._combine(other, sign=-1)

    def __pow__(self, power: float) -> "Dimension":
        result = Dimension()
        result._powers = {u: p * power for u, p in self._powers.items()}
        result._clean()
        return result


# ----------------------------------------------------------------------
# Standard NMR‑related dimensions
# ----------------------------------------------------------------------

def _hz(exp: float = 1.0) -> Dimension:
    return Dimension(Hz=exp)


DIMENSIONS: Dict[str, Dimension] = {
    # Physical (frequency) variables
    "j": _hz(1),   # J_intra (Hz)
    "dJ": _hz(1),  # ΔJ (Hz)

    # Dimensionless ratios / constants
    "dJ/j": Dimension(),
    "j/dJ": Dimension(),
    "pi": Dimension(),
    "e": Dimension(),
    "2": Dimension(),
    "0.5": Dimension(),
    "sqrt(2)": Dimension(),
}


# ----------------------------------------------------------------------
# Helpers to provide dimensions for *symbolic* variables in formulas
# ----------------------------------------------------------------------

def get_variable_dimensions() -> Dict[str, Dimension]:
    """Return a mapping **variable‑name → Dimension** for model inputs."""
    return {
        # x0 = ΔJ, x1 = ΔJ/J_intra, x2 = J_intra, x3 = J_intra/ΔJ
        "x_0": _hz(1),
        "x_1": Dimension(),
        "x_2": _hz(1),
        "x_3": Dimension(),

        # Aliases matching the raw physics notation
        "j": _hz(1),
        "dJ": _hz(1),
    }


# ----------------------------------------------------------------------
# Core dimensional analysis engine
# ----------------------------------------------------------------------

def get_expression_dimension(expr: sp.Expr, var_dims: Dict[str, Dimension]) -> Dimension:
    """Recursively evaluate the dimension of a SymPy expression."""

    # (1) Literals -----------------------------------------------------------------
    if expr.is_number:
        return Dimension()  # pure numbers → dimensionless

    if expr.is_symbol:
        return var_dims.get(str(expr), Dimension())

    # (2) Addition -----------------------------------------------------------------
    if expr.is_Add:
        term_dims = [get_expression_dimension(t, var_dims) for t in expr.args]
        if not term_dims:
            return Dimension()
        first = term_dims[0]
        for other in term_dims[1:]:
            if other != first:
                raise ValueError(
                    f"Dimensional inconsistency in sum: {first} vs {other} in {expr}"  # noqa: E501
                )
        return first

    # (3) Multiplication -----------------------------------------------------------
    if expr.is_Mul:
        result = Dimension()
        for factor in expr.args:
            result = result * get_expression_dimension(factor, var_dims)
        return result

    # (4) Powers -------------------------------------------------------------------
    if expr.is_Pow:
        base, exponent = expr.args
        base_dim = get_expression_dimension(base, var_dims)
        exp_dim = get_expression_dimension(exponent, var_dims)
        if not exp_dim.is_dimensionless():
            raise ValueError(f"Exponent must be dimensionless, got {exp_dim} in {expr}")
        if exponent.is_number:
            return base_dim ** float(exponent)
        # Symbolic exponent (dimensionless by check above) → keep same dimension
        return base_dim

    # (5) Special funcs requiring dimensionless args --------------------------------
    if expr.func in {sp.sin, sp.cos, sp.tan, sp.exp, sp.log}:
        arg_dim = get_expression_dimension(expr.args[0], var_dims)
        if not arg_dim.is_dimensionless():
            raise ValueError(
                f"Argument to {expr.func.__name__} must be dimensionless, got {arg_dim} in {expr}"
            )
        # sin/cos/… return dimensionless
        return Dimension()

    # (6) sqrt ---------------------------------------------------------------------
    if expr.func == sp.sqrt:
        return get_expression_dimension(expr.args[0], var_dims) ** 0.5

    # (7) Default fallback: assume dimension of first argument ----------------------
    if expr.args:
        return get_expression_dimension(expr.args[0], var_dims)

    return Dimension()


# ----------------------------------------------------------------------
# Consistency check utilities
# ----------------------------------------------------------------------

def check_dimensional_consistency(
    expr: sp.Expr,
    expected_output_dim: Dimension,
    var_dims: Optional[Dict[str, Dimension]] = None,
) -> Tuple[bool, str]:
    if var_dims is None:
        var_dims = get_variable_dimensions()
    try:
        actual_dim = get_expression_dimension(expr, var_dims)
        if actual_dim == expected_output_dim:
            return True, "Dimensionally consistent"
        return False, f"Expected {expected_output_dim}, got {actual_dim}"
    except ValueError as err:
        return False, str(err)


# ----------------------------------------------------------------------
# Automatic *fix‑ups* for poorly formed formulas
# ----------------------------------------------------------------------

def fix_exponential_dimensions(
    expr: sp.Expr, var_dims: Optional[Dict[str, Dimension]] = None
) -> sp.Expr:
    if var_dims is None:
        var_dims = get_variable_dimensions()

    def _fix(e):
        if e.func == sp.exp:
            arg = e.args[0]
            dim = get_expression_dimension(arg, var_dims)
            if not dim.is_dimensionless():
                # Replace exp(bad) with 0 to flag issue (could also raise)
                return sp.S.Zero
        return e

    return expr.replace(lambda z: z.func == sp.exp, _fix)


def fix_sum_dimensions(
    expr: sp.Expr, var_dims: Optional[Dict[str, Dimension]] = None
) -> sp.Expr:
    if var_dims is None:
        var_dims = get_variable_dimensions()

    def _fix(e):
        if not e.is_Add:
            return e
        groups: Dict[Tuple[Tuple[str, float], ...], list] = {}
        for term in e.args:
            try:
                dim = get_expression_dimension(term, var_dims)
                key = tuple(sorted(dim._powers.items()))  # hashable key
            except ValueError:
                key = (("<unknown>", id(term)),)
            groups.setdefault(key, []).append(term)
        # Keep the largest (most terms) group; drop the rest
        best_terms = groups.get((('Hz', 1.0),), [])
        return sp.Add(*best_terms) if len(best_terms) > 1 else best_terms[0]

    return expr.replace(lambda z: z.is_Add, _fix)


def apply_dimensional_analysis(
    expr: sp.Expr,
    expected_output_dim: Dimension,
    var_dims: Optional[Dict[str, Dimension]] = None,
):
    if var_dims is None:
        var_dims = get_variable_dimensions()

    expr1 = fix_exponential_dimensions(expr, var_dims)
    expr2 = fix_sum_dimensions(expr1, var_dims)
    ok, msg = check_dimensional_consistency(expr2, expected_output_dim, var_dims)
    return expr2, ok, msg


# ----------------------------------------------------------------------
# NMR‑specific convenience wrappers
# ----------------------------------------------------------------------

def get_nmr_output_dimension() -> Dimension:
    return _hz(1)


def analyze_nmr_formula(
    formula: sp.Expr, 
    var_dims: Optional[Dict[str, Dimension]] = None,
    expected_output_dim: Optional[Dimension] = None
) -> Dict:
    """
    Analyze and fix dimensional consistency of an NMR formula.
    
    Args:
        formula: SymPy expression to analyze
        var_dims: Variable dimensions mapping (defaults to NMR variables)
        expected_output_dim: Expected output dimension (defaults to Hz)
    
    Returns:
        Dictionary with analysis results
    """
    if var_dims is None:
        var_dims = get_variable_dimensions()
    
    if expected_output_dim is None:
        expected_output_dim = get_nmr_output_dimension()

    ok0, msg0 = check_dimensional_consistency(formula, expected_output_dim, var_dims)
    fixed, ok1, msg1 = apply_dimensional_analysis(formula, expected_output_dim, var_dims)

    return {
        "original_formula": formula,
        "fixed_formula": fixed,
        "expected_dimension": expected_output_dim,
        "original_consistent": ok0,
        "original_message": msg0,
        "fixed_consistent": ok1,
        "fixed_message": msg1,
        "formula_changed": not formula.equals(fixed),
    }


def optimize_formula_coefficients(
    formula: sp.Expr,
    x_data: np.ndarray,
    y_data: np.ndarray,
    max_epochs: int = 1000,
    lr: float = 0.01,
    tolerance: float = 1e-6
) -> Dict:
    """
    Optimize coefficients in a formula using torch-based gradient descent.
    
    Args:
        formula: SymPy expression with coefficients to optimize
        x_data: Input data as numpy array (samples x features)
        y_data: Target data as numpy array (samples,)
        max_epochs: Maximum training epochs
        lr: Learning rate
        tolerance: Early stopping tolerance
    
    Returns:
        Dictionary with optimization results
    """
    try:
        from .sympy_torch import torchModelFromFormula
    except ImportError:
        return {
            "optimized_formula": formula,
            "original_mse": float('inf'),
            "optimized_mse": float('inf'),
            "optimization_success": False,
            "message": "torchModelFromFormula not available"
        }
    
    try:
        # Convert physics variable names back to x_0, x_1, etc. for torch compatibility
        j, dJ = sp.symbols("j dJ")
        x0, x1, x2, x3 = sp.symbols("x_0 x_1 x_2 x_3")
        
        # Create comprehensive mapping based on the number of input features
        n_features = x_data.shape[1]
        
        # First, substitute all instances of j and dJ symbols
        # This handles complex expressions like dJ**2/j**2, dJ/j**1.0, etc.
        if n_features == 2:
            # x0=deltaJ, x1=deltaJ/Jintra
            # For 2D, we need to be careful since we don't have j directly
            # We'll substitute dJ with x0, and any dJ/j ratios with x1
            torch_formula = formula.subs(dJ/j, x1)  # First substitute ratios
            torch_formula = torch_formula.subs(dJ, x0)  # Then substitute dJ
            # Any remaining j should be expressed in terms of dJ and dJ/j
            # j = dJ / (dJ/j) = dJ / x1 = x0 / x1
            torch_formula = torch_formula.subs(j, x0/x1)
        elif n_features == 3:
            # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra
            # Substitute in order: ratios first, then individual variables
            torch_formula = formula.subs(dJ/j, x1)  # First substitute ratios
            torch_formula = torch_formula.subs(j/dJ, 1/x1)  # Handle inverse ratios
            torch_formula = torch_formula.subs(dJ, x0)  # Then substitute dJ
            torch_formula = torch_formula.subs(j, x2)   # Finally substitute j
        elif n_features == 4:
            # x0=deltaJ, x1=deltaJ/Jintra, x2=Jintra, x3=Jintra/deltaJ
            torch_formula = formula.subs(dJ/j, x1)   # First substitute ratios
            torch_formula = torch_formula.subs(j/dJ, x3)  # Handle inverse ratios
            torch_formula = torch_formula.subs(dJ, x0)   # Then substitute dJ
            torch_formula = torch_formula.subs(j, x2)    # Finally substitute j
        else:
            # Fallback: try to map any remaining physics symbols
            torch_formula = formula.subs(dJ/j, x1)  # Handle ratios first
            torch_formula = torch_formula.subs(dJ, x0)
            torch_formula = torch_formula.subs(j, x2)
        
        # Simplify the torch formula
        torch_formula = sp.simplify(torch_formula)
        
        # Convert numpy data to torch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare input data dictionary
        input_dict = {}
        for i in range(n_features):
            input_dict[f'x_{i}'] = torch.tensor(
                x_data[:, i:i+1], dtype=torch.float32, device=device
            )
        
        y_target = torch.tensor(y_data, dtype=torch.float32, device=device)
        
        # Create torch model from formula
        model = torchModelFromFormula(torch_formula).to(device)
        
        # Calculate original MSE
        with torch.no_grad():
            y_pred_orig = model(input_dict).squeeze()
            original_mse = torch.nn.functional.mse_loss(y_pred_orig, y_target).item()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        max_patience = 50
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            
            y_pred = model(input_dict).squeeze()
            loss = criterion(y_pred, y_target)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            # Early stopping
            if current_loss < best_loss - tolerance:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        # Get optimized formula and convert back to physics variable names
        optimized_torch_formula = model.to_sympy()
        
        # Convert back to physics names for consistency
        # Use the reverse mapping, handling complex expressions
        if n_features == 2:
            # Reverse: x0 → dJ, x1 → dJ/j, x0/x1 → j
            optimized_formula = optimized_torch_formula.subs(x0/x1, j)  # Handle j expressions first
            optimized_formula = optimized_formula.subs(x1, dJ/j)       # Then ratios
            optimized_formula = optimized_formula.subs(x0, dJ)         # Finally dJ
        elif n_features == 3:
            # Reverse: x0 → dJ, x1 → dJ/j, x2 → j, 1/x1 → j/dJ
            optimized_formula = optimized_torch_formula.subs(1/x1, j/dJ)  # Handle inverse ratios first
            optimized_formula = optimized_formula.subs(x1, dJ/j)         # Then ratios
            optimized_formula = optimized_formula.subs(x0, dJ)           # Then dJ
            optimized_formula = optimized_formula.subs(x2, j)            # Finally j
        elif n_features == 4:
            # Reverse: x0 → dJ, x1 → dJ/j, x2 → j, x3 → j/dJ
            optimized_formula = optimized_torch_formula.subs(x3, j/dJ)   # Handle inverse ratios first
            optimized_formula = optimized_formula.subs(x1, dJ/j)        # Then ratios
            optimized_formula = optimized_formula.subs(x0, dJ)          # Then dJ
            optimized_formula = optimized_formula.subs(x2, j)           # Finally j
        else:
            # Fallback
            optimized_formula = optimized_torch_formula.subs(x1, dJ/j)
            optimized_formula = optimized_formula.subs(x0, dJ)
            optimized_formula = optimized_formula.subs(x2, j)
        
        # Simplify the final formula
        optimized_formula = sp.simplify(optimized_formula)
        
        final_mse = best_loss
        
        return {
            "optimized_formula": optimized_formula,
            "original_mse": original_mse,
            "optimized_mse": final_mse,
            "optimization_success": True,
            "epochs_trained": epoch + 1,
            "message": f"Optimization completed in {epoch + 1} epochs"
        }
        
    except Exception as e:
        return {
            "optimized_formula": formula,
            "original_mse": float('inf'),
            "optimized_mse": float('inf'),
            "optimization_success": False,
            "message": f"Optimization failed: {str(e)}"
        }
