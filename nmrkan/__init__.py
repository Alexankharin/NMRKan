"""Core package for NMR symbolic regression."""
from .models import KharKAN, DenseKanLayer, _clean_expr
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
    "KharKAN",
    "DenseKanLayer",
    "_clean_expr",
    "get_frequences",
    "get_frequences_ordered",
    "ham_reduced_symmetric",
    "ham_reduced_antisymmetric",
    "HMatrixIdealized",
    "CalcFreqsIdealized",
    "compute_transition_frequency",
    "Dimension",
    "analyze_nmr_formula",
    "check_dimensional_consistency",
    "apply_dimensional_analysis",
    "get_variable_dimensions",
    "get_nmr_output_dimension",
    "optimize_formula_coefficients",
]
