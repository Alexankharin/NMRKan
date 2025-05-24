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

# ———————— user-specified constants ————————
JAA       = 14.0
dJAM      =   6.06
dJMX      =   6.06
dJAX      =   0.0

# ranges for the “swept” parameters
dJMM_min, dJMM_max, ddJMM = -0.1, +0.1, 0.005
dJXX_min, dJXX_max, ddJXX = -0.1, +0.1, 0.005

dJMM_vals = np.arange(dJMM_min, dJMM_max + ddJMM/2, ddJMM)
dJXX_vals = np.arange(dJXX_min, dJXX_max + ddJXX/2, ddJXX)

# preallocate lists
records: List[Dict[str, Any]] = []

for dJMM in dJMM_vals:
    for dJXX in dJXX_vals:
        # build Hamiltonians
        H_sym  = ham_reduced_symmetric(JAA, dJMM, dJXX, dJAM, dJMX, dJAX)
        H_anti = ham_reduced_antisymmetric(JAA, dJMM, dJXX, dJAM, dJMX, dJAX)

        # eigenvalues
        w_sym, _  = eigh(H_sym)
        w_anti, _ = eigh(H_anti)

        # extract the first nonzero transition frequency
        f_sym  = compute_transition_frequency(w_sym)
        f_anti = compute_transition_frequency(w_anti)

        # store one row
        records.append({
            "dJMM":      dJMM,
            "dJXX":      dJXX,
            "FreqSym":   f_sym,
            "FreqAnti":  f_anti
        })

# build a DataFrame and write out
df = pd.DataFrame.from_records(records)
df.to_csv("KAN_dataset_reduced.csv", index=False)

print(f"Saved {len(df)} samples to KAN_dataset_reduced.csv")
