"""Core package for NMR symbolic regression."""
from .models import KharKAN, DenseKanLayer, _clean_expr
from .nmr import get_frequences, get_frequences_ordered
from .datagen import (
    ham_reduced_symmetric,
    ham_reduced_antisymmetric,
    HMatrixIdealized,
    CalcFreqsIdealized,
    compute_transition_frequency,
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
]
