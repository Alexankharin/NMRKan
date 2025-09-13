"""Data generation package for NMR symbolic regression experiments."""

from .eigenvalues_4CH2 import (
    Ham4CH2s3Sand1S,
    H4CH2sMatrixIdealized,
    CalcEigenEnergies,
)

from .nmr_datagen import (
    ham_reduced_symmetric,
    ham_reduced_antisymmetric,
    HMatrixIdealized,
    CalcFreqsIdealized,
    CalcFreqsIdealized as CalcFreqsFromHMatrix,
    compute_transition_frequency,
    get_frequences,
    get_frequences_ordered,
)

from .datasets import (
    make_dataset_from_function,
    make_dimensionless_dataset_from_function,
    get_eigenvalue_dimensionless,
    get_perturbation_dimensionless,
    get_perturbation,
    get_normalized_eigenvalues,
    make_normalized_eigenvalue_dataset,
    NormalizedEigenvalueSurface,
)

__all__ = [
    # 4CH2 specific functions
    "Ham4CH2s3Sand1S",
    "H4CH2sMatrixIdealized", 
    "CalcEigenEnergies",
    "CalcFreqsIdealized",
    # General NMR functions
    "ham_reduced_symmetric",
    "ham_reduced_antisymmetric",
    "HMatrixIdealized",
    "CalcFreqsFromHMatrix",
    "compute_transition_frequency",
    "get_frequences",
    "get_frequences_ordered",
    # Dataset generation
    "make_dataset_from_function",
    "make_dimensionless_dataset_from_function",
    "get_eigenvalue_dimensionless",
    "get_perturbation_dimensionless",
    "get_perturbation",
    "get_normalized_eigenvalues",
    "make_normalized_eigenvalue_dataset",
    "NormalizedEigenvalueSurface",
]