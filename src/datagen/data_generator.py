import numpy as np
import pandas as pd
from scipy.linalg import eigh
from typing import Any, List, Dict

def ham_reduced_symmetric(
    JAA: float, JMM: float, JXX: float, dJAM: float, dJMX: float, dJAX: float
) -> np.ndarray:
    """
    Construct the 4×4 symmetric subspace Hamiltonian.

    Args:
        JAA (float): Coupling constant.
        JMM (float): Coupling constant.
        JXX (float): Coupling constant.
        dJAM (float): Coupling constant.
        dJMX (float): Coupling constant.
        dJAX (float): Coupling constant.

    Returns:
        np.ndarray: 4x4 symmetric Hamiltonian matrix.
    """
    H = np.array([
        [ 1/4*(-3*JAA - 3*JMM + JXX),   dJMX/2,                 dJAX/2,                dJAM/2  ],
        [ dJMX/2,                      1/4*(-3*JAA + JMM - 3*JXX), dJAM/2,                    dJAX/2  ],
        [  dJAX/2,                      dJAM/2,                    1/4*(JAA - 3* JMM -3* JXX), dJMX/2  ],
        [ dJAM/2,                           dJAX/2,                 dJMX/2,                 1/4*(JAA + JMM + JXX) ]
    ], dtype=float)
    return H

def ham_reduced_antisymmetric(
    JAA: float, JMM: float, JXX: float, dJAM: float, dJMX: float, dJAX: float
) -> np.ndarray:
    """
    Construct the 4×4 antisymmetric subspace Hamiltonian.

    Args:
        JAA (float): Coupling constant.
        JMM (float): Coupling constant.
        JXX (float): Coupling constant.
        dJAM (float): Coupling constant.
        dJMX (float): Coupling constant.
        dJAX (float): Coupling constant.

    Returns:
        np.ndarray: 4x4 antisymmetric Hamiltonian matrix.
    """
    H = np.array([
        [ 1/4*(-3*JAA - 3*JMM + JXX),  dJMX/2,                 dJAX/2,                 0        ],
        [ dJAX/2,                      1/4*( JMM + JXX + JAA),  -JMM,                   dJMX/2  ],
        [ dJAM/2,                     -JMM,                     1/4*( JMM + JXX + JAA), dJAX/2  ],
        [ 0,                           dJMX/2,                 dJAX/2,                 1/4*(JAA - 3*JMM - 3*JXX) ]
    ], dtype=float)
    return H


def HMatrixIdealized(Jintra, deltaJ):
    """
    The ‘idealized’ Hamiltonian used in the NB:
      JAA = JMM = JXX = Jintra
      dJAM = dJMX = deltaJ,   dJAX = 0
    """
    return ham_reduced_symmetric(
        JAA=Jintra,
        JMM=Jintra,
        JXX=Jintra,
        dJAM=deltaJ,
        dJMX=deltaJ,
        dJAX=0.0
    )

def CalcFreqsIdealized(Jintra, DeltaJ):
    """
    Calculate characteristic frequencies from the Hamiltonian matrix
    constructed using Jintra and DeltaJ.

    Parameters:
    Jintra (float): Intra-coupling constant.
    DeltaJ (float): Coupling asymmetry or perturbation.

    Returns:
    list: Frequencies [f1, f2, f3] computed from energy differences.
    """

    # Replace this with the actual function to get your 4x4 matrix
    TM = HMatrixIdealized(Jintra, DeltaJ)

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(TM)

    # Define reference states
    state1 = 0.5 * np.array([1, np.sqrt(2), 1, 0])
    state2 = 0.5 * np.array([np.sqrt(2), 0, -np.sqrt(2), 0])
    state3 = 0.5 * np.array([1, -np.sqrt(2), 1, 0])

    # Compute scalar products with each eigenvector
    SP1 = np.array([np.dot(state1, evecs[:, i]) for i in range(4)])
    SP2 = np.array([np.dot(state2, evecs[:, i]) for i in range(4)])
    SP3 = np.array([np.dot(state3, evecs[:, i]) for i in range(4)])

    # Find indices of maximum overlap
    N1 = np.argmax(np.abs(SP1))
    N2 = np.argmax(np.abs(SP2))
    N3 = np.argmax(np.abs(SP3))

    # Get corresponding energies
    En1 = evals[N1]
    En2 = evals[N2]
    En3 = evals[N3]

    # Compute frequencies
    f1 = abs(En2 - En3)
    f2 = abs(En1 - En2)
    f3 = abs(En1 - En3)

    freqlist = [f1, f2, f3]

    return freqlist

def compute_transition_frequency(eigvals: np.ndarray) -> float:
    """
    Given a sorted array of 4 eigenvalues, return the smallest non-zero
    gap between successive levels (i.e. the first “transition frequency”).

    Args:
        eigvals (np.ndarray): Array of eigenvalues.

    Returns:
        float: The smallest non-zero transition frequency, or 0.0 if none found.
    """
    # ensure sorted
    w = np.sort(eigvals)
    # compute all differences
    diffs = np.diff(w)
    # ignore any zero gaps (degeneracies) and take the first positive
    freqs = diffs[diffs > 1e-8]
    return freqs[0] if len(freqs) else 0.0