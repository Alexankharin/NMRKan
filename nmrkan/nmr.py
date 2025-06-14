import numpy as np
from scipy.linalg import eigh

from .datagen import ham_reduced_symmetric, CalcFreqsIdealized

__all__ = [
    "get_ham_simple",
    "get_levels",
    "get_frequences",
    "get_frequences_ordered",
]


def get_ham_simple(Jintra: float, deltaJ: float) -> np.ndarray:
    """Return the Hamiltonian matrix for a simple two-parameter system."""
    return ham_reduced_symmetric(Jintra, Jintra, Jintra, deltaJ, deltaJ, 0.0)


def get_levels(matrix: np.ndarray) -> list:
    """Return all energy level differences for ``matrix``."""
    evals, _ = eigh(matrix)
    levels = evals[evals > 1e-8]
    levels.sort()
    energies = [levels[j] - levels[i] for i in range(len(levels)) for j in range(i + 1, len(levels))]
    return energies


def get_frequences(Jintra: np.ndarray, deltaJ: np.ndarray | None = None) -> list:
    """Compute transition frequencies for each pair of ``Jintra`` and ``deltaJ``."""
    if deltaJ is None:
        Jintra = np.asarray(Jintra)
        if Jintra.ndim == 2 and Jintra.shape[1] == 2:
            Jintra_arr, deltaJ_arr = Jintra[:, 0], Jintra[:, 1]
        else:
            raise ValueError("If deltaJ is None, Jintra must have shape (B,2)")
    else:
        Jintra_arr = np.asarray(Jintra)
        deltaJ_arr = np.asarray(deltaJ)
        if Jintra_arr.shape != deltaJ_arr.shape:
            raise ValueError("Jintra and deltaJ must have the same shape")

    results = []
    for j, d in zip(Jintra_arr, deltaJ_arr):
        matrix = get_ham_simple(j, d)
        results.append(get_levels(matrix))
    return results


def get_frequences_ordered(Jintra: np.ndarray, deltaJ: np.ndarray | None = None) -> list:
    """Same as :func:`get_frequences` but returns sorted frequencies."""
    if deltaJ is None:
        Jintra = np.asarray(Jintra)
        if Jintra.ndim == 2 and Jintra.shape[1] == 2:
            Jintra_arr, deltaJ_arr = Jintra[:, 0], Jintra[:, 1]
        else:
            raise ValueError("If deltaJ is None, Jintra must have shape (B,2)")
    else:
        Jintra_arr = np.asarray(Jintra)
        deltaJ_arr = np.asarray(deltaJ)
        if Jintra_arr.shape != deltaJ_arr.shape:
            raise ValueError("Jintra and deltaJ must have the same shape")

    results = []
    for j, d in zip(Jintra_arr, deltaJ_arr):
        freqs = CalcFreqsIdealized(j, d)
        results.append(sorted(freqs))
    return results
