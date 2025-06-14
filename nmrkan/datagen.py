import numpy as np
from scipy.linalg import eigh

__all__ = [
    "ham_reduced_symmetric",
    "ham_reduced_antisymmetric",
    "HMatrixIdealized",
    "CalcFreqsIdealized",
    "compute_transition_frequency",
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
