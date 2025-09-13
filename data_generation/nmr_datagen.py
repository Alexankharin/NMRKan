import numpy as np
from scipy.linalg import eigh

__all__ = [
    "ham_reduced_symmetric",
    "ham_reduced_antisymmetric",
    "HMatrixIdealized",
    "CalcFreqsIdealized",
    "compute_transition_frequency",
    "get_ham_simple",
    "get_levels",
    "get_frequences",
    "get_frequences_ordered",
]


def ham_reduced_symmetric(JAA: float, JMM: float, JXX: float,
                          dJAM: float, dJMX: float, dJAX: float) -> np.ndarray:
    """Return the 4x4 symmetric subspace Hamiltonian."""
    return np.array([
        [0.25 * (-3 * JAA - 3 * JMM + JXX),  dJMX / 2,               dJAX / 2, dJAM / 2],
        [dJMX / 2,                          0.25 * (-3 * JAA + JMM - 3 * JXX), dJAM / 2, dJAX / 2],
        [dJAX / 2,                          dJAM / 2,              0.25 * (JAA - 3 * JMM - 3 * JXX), dJMX / 2],
        [dJAM / 2,                          dJAX / 2,               dJMX / 2, 0.25 * (JAA + JMM + JXX)],
    ], dtype=float)


def ham_reduced_antisymmetric(JAA: float, JMM: float, JXX: float,
                              dJAM: float, dJMX: float, dJAX: float) -> np.ndarray:
    """Return the 4x4 antisymmetric subspace Hamiltonian."""
    return np.array([
        [0.25 * (-3 * JAA - 3 * JMM + JXX),  dJMX / 2,               dJAX / 2, 0.0],
        [dJAX / 2,                          0.25 * (JMM + JXX + JAA), -JMM,      dJMX / 2],
        [dJAM / 2,                         -JMM,                      0.25 * (JMM + JXX + JAA), dJAX / 2],
        [0.0,                               dJMX / 2,               dJAX / 2, 0.25 * (JAA - 3 * JMM - 3 * JXX)],
    ], dtype=float)


def HMatrixIdealized(Jintra: float, deltaJ: float) -> np.ndarray:
    """Convenience wrapper used in the notebooks."""
    return ham_reduced_symmetric(Jintra, Jintra, Jintra, deltaJ, deltaJ, 0.0)


def CalcFreqsIdealized(Jintra: float, deltaJ: float) -> list:
    """Calculate characteristic frequencies from ``Jintra`` and ``deltaJ``."""
    TM = HMatrixIdealized(Jintra, deltaJ)
    evals, evecs = eigh(TM)

    state1 = 0.5 * np.array([1, np.sqrt(2), 1, 0])
    state2 = 0.5 * np.array([np.sqrt(2), 0, -np.sqrt(2), 0])
    state3 = 0.5 * np.array([1, -np.sqrt(2), 1, 0])

    SP1 = np.array([np.dot(state1, evecs[:, i]) for i in range(4)])
    SP2 = np.array([np.dot(state2, evecs[:, i]) for i in range(4)])
    SP3 = np.array([np.dot(state3, evecs[:, i]) for i in range(4)])

    N1 = int(np.argmax(np.abs(SP1)))
    N2 = int(np.argmax(np.abs(SP2)))
    N3 = int(np.argmax(np.abs(SP3)))

    En1, En2, En3 = evals[N1], evals[N2], evals[N3]
    return [abs(En2 - En3), abs(En1 - En2), abs(En1 - En3)]


def compute_transition_frequency(eigvals: np.ndarray) -> float:
    """Return the smallest non-zero gap between successive eigenvalues."""
    w = np.sort(eigvals)
    diffs = np.diff(w)
    freqs = diffs[diffs > 1e-8]
    return float(freqs[0]) if len(freqs) else 0.0


def get_ham_simple(Jintra: float, deltaJ: float) -> np.ndarray:
    """Return the Hamiltonian matrix for a simple two-parameter system."""
    return ham_reduced_symmetric(Jintra, Jintra, Jintra, deltaJ, deltaJ, 0.0)


def get_levels(matrix: np.ndarray) -> list[float]:
    """Return all energy level differences for ``matrix`` (three non-zero transitions)."""
    evals, _ = eigh(matrix)
    # filter out (near) zero ground state
    levels = np.sort(evals[evals > 1e-8])
    if levels.size != 3:
        raise ValueError(f"Expected 3 non-zero levels, got {levels.size}")
    # transition frequencies between pairs: (1-0, 2-0, 2-1)
    return [levels[1] - levels[0], levels[2] - levels[0], levels[2] - levels[1]]


def _parse_inputs(Jintra: np.ndarray | list, deltaJ: np.ndarray | list | None):
    """Internal: parse Jintra and deltaJ into flat 1D arrays of equal length B."""
    if deltaJ is None:
        arr = np.asarray(Jintra)
        if arr.ndim == 2 and arr.shape[1] == 2:
            Jintra_arr = arr[:, 0]
            deltaJ_arr = arr[:, 1]
        else:
            raise ValueError("If deltaJ is None, Jintra must have shape (B,2)")
    else:
        Jintra_arr = np.asarray(Jintra)
        deltaJ_arr = np.asarray(deltaJ)
        if Jintra_arr.shape != deltaJ_arr.shape:
            raise ValueError("Jintra and deltaJ must have the same shape")
    # flatten to 1D
    Jintra_arr = Jintra_arr.flatten()
    deltaJ_arr = deltaJ_arr.flatten()
    return Jintra_arr, deltaJ_arr


def get_frequences(
    Jintra: np.ndarray | list, deltaJ: np.ndarray | list | None = None
) -> np.ndarray:
    """Compute transition frequencies for each pair of ``Jintra`` and ``deltaJ``.
    Returns an array of shape (B, 3)."""
    Jintra_arr, deltaJ_arr = _parse_inputs(Jintra, deltaJ)
    B = Jintra_arr.size
    freqs = np.zeros((B, 3), dtype=float)
    for idx, (j, d) in enumerate(zip(Jintra_arr, deltaJ_arr)):
        vals = get_levels(get_ham_simple(j, d))
        freqs[idx, :] = vals
    return freqs


def get_frequences_ordered(
    Jintra: np.ndarray | list, deltaJ: np.ndarray | list | None = None
) -> np.ndarray:
    """Same as :func:`get_frequences` but returns sorted frequencies.
    Inputs parsed as in get_frequences; returns array of shape (B, 3)."""
    Jintra_arr, deltaJ_arr = _parse_inputs(Jintra, deltaJ)
    B = Jintra_arr.size
    freqs = np.zeros((B, 3), dtype=float)
    for idx, (j, d) in enumerate(zip(Jintra_arr, deltaJ_arr)):
        vals = np.asarray(CalcFreqsIdealized(j, d), dtype=float)
        if vals.size != 3:
            raise ValueError(
                f"Expected 3 frequencies from CalcFreqsIdealized, got {vals.size}"
            )
        freqs[idx, :] = np.sort(vals)
    return freqs