import numpy as np

from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

def Ham4CH2s3Sand1S(Jintra: float, DeltaJ: float) -> np.ndarray:
    """Build the 8×8 Hamiltonian from Jintra and ΔJ (Mathematica: Ham4CH2s3Sand1S)."""
    d = DeltaJ / 2.0
    m = -2.0 * Jintra
    H = np.array([
        [m,  d, 0, 0, 0,  d, 0,  d],
        [d,  m, d, 0, d,  0, 0,  0 ],
        [0,  d, m, d, 0,  0, 0,  d ],
        [0,  0, d, m, d,  0, d,  0 ],
        [0,  d, 0, d, 0,  d, 0,  0 ],
        [d,  0, 0, 0, d,  0, d,  0 ],
        [0,  0, 0, d, 0,  d, 0,  d ],
        [d,  0, d, 0, 0,  0, d,  0 ],
    ], dtype=float)
    return H


def H4CH2sMatrixIdealized(x1: float, x2: float) -> np.ndarray:
    """Mathematica: H4CH2sMatrixIdealized[x1_, x2_] := Ham4CH2s3Sand1S[x1, x2]."""
    return Ham4CH2s3Sand1S(x1, x2)


def _coeff_matrix(n: int) -> np.ndarray:
    """Build U (size n×n) with elements sqrt(2/(n+1)) * sin((j*k*pi)/(n+1)), j,k=1..n."""
    idx = np.arange(1, n + 1, dtype=float)
    return np.sqrt(2.0 / (n + 1.0)) * np.sin(np.pi * np.outer(idx, idx) / (n + 1.0))


def CalcEigenEnergies(Jintra: float, DeltaJ: float):
    """
    Python translation of your Mathematica CalcEigenEnergies.
    Returns:
      States                -> 8×8 array (U3Sand1S)
      ScalarProductMatrix   -> 8×8 array; entry (i,j) = States[j]·EigenBasis[i]
      NumberOfEigenState    -> length-8 array (1-based indices, to match Mathematica)
      Energies              -> length-8 array; eigenvalues picked via the above indices
      Eigensystem_TM        -> (eigenvalues, eigenvectors) where eigenvectors are columns
    """
    TM = H4CH2sMatrixIdealized(Jintra, DeltaJ)

    # Symmetric, so use eigh (sorted ascending); eigvecs are columns
    eigenvals, eigvecs = np.linalg.eigh(TM)

    # Build U3S (n=4) and block-diagonal U3S ⊕ U3S
    NspinPairs = 4
    U3S = _coeff_matrix(NspinPairs)

    States = np.zeros((2 * NspinPairs, 2 * NspinPairs), dtype=float)
    States[:NspinPairs, :NspinPairs] = U3S
    States[NspinPairs:, NspinPairs:] = U3S  # bottom-right block

    # ScalarProductMatrix[i, j] = States[j] · EigenBasis[i]
    # With eigvecs as columns: SPM = eigvecs.T @ States.T
    ScalarProductMatrix = eigvecs.T @ States.T  # shape (8,8)

    # For each column j (state j), find i with maximal |SPM[i, j]|
    max_rows_per_col = np.argmax(np.abs(ScalarProductMatrix), axis=0)  # 0-based
    NumberOfEigenState_0based = max_rows_per_col
    Energies = eigenvals[NumberOfEigenState_0based]

    return Energies


def _match_by_overlap(prev_V: np.ndarray, curr_V: np.ndarray):
    """
    Match eigenvectors in curr_V to prev_V maximizing |<prev_i | curr_j>|.
    Also fixes arbitrary phases so overlaps are positive real.

    Args:
        prev_V: (n, n) eigenvectors (columns) at previous grid point
        curr_V: (n, n) eigenvectors (columns) at current grid point (unsorted)

    Returns:
        perm:   permutation indices to apply to eigenvalues of current step
        V_mat:  reordered & phase-aligned eigenvectors for current step
    """
    # Overlap matrix M_ij = |<prev_i | curr_j>|
    M = np.abs(prev_V.conj().T @ curr_V)  # (n, n)
    # Hungarian assignment on cost = 1 - overlap
    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)

    # Reorder current eigenvectors
    V_perm = curr_V[:, col_ind]

    # Phase alignment: make <prev_i | curr_i> real & positive
    phase = np.einsum('ij,ij->j', prev_V.conj(), V_perm)  # inner products for each matched pair
    denom = np.where(np.abs(phase) > 0, np.abs(phase), 1.0)
    phase = phase / denom
    V_perm = V_perm * phase  # broadcast over columns

    return col_ind, V_perm


def get_surfaces_vectors_in_region(
    min_Jintra: float,
    max_Jintra: float,
    min_deltaJ: float,
    max_deltaJ: float,
    deltaJ_points: int = 200,
    jintra_points: int = 200,
):
    """
    Track & return smoothly ordered eigenvalue/eigenvector surfaces for an 8x8 Hermitian
    Hamiltonian H = Ham4CH2s3Sand1S(J_intra, DeltaJ) over a 2D parameter grid.

    Grid axes:
        axis-0: DeltaJ (size dJStep)
        axis-1: J_intra (size Jistep)

    Returns:
        deltaJ_grid: 1D array shape (dJStep,)
        Jintra_grid: 1D array shape (Jistep,)
        eigvals:     array shape (dJStep, Jistep, 8)         (tracked order)
        eigvecs:     array shape (dJStep, Jistep, 8, 8)      (columns are eigenvectors)
    """
    # Build parameter grids
    deltaJ_grid = np.linspace(min_deltaJ, max_deltaJ, deltaJ_points)
    Jintra_grid = np.linspace(min_Jintra, max_Jintra, jintra_points)

    n = 8
    eigvals = np.empty((deltaJ_points, jintra_points, n), dtype=float)
    eigvecs = np.empty((deltaJ_points, jintra_points, n, n), dtype=np.complex128)

    # --- Seed at (0,0)
    jintra0 = Jintra_grid[0]
    deltaJ0 = deltaJ_grid[0]
    A0 = Ham4CH2s3Sand1S(jintra0, deltaJ0)  # expected Hermitian 8x8
    w0, V0 = eigh(A0)  # returns sorted ascending by default for Hermitian
    eigvals[0, 0] = w0
    eigvecs[0, 0] = V0

    # Sweep the grid in row-major order:
    for i in range(deltaJ_points):
        for j in range(jintra_points):
            if i == 0 and j == 0:
                continue

            jintra = Jintra_grid[j]
            deltaJ = deltaJ_grid[i]
            A = Ham4CH2s3Sand1S(jintra, deltaJ)
            w, V = eigh(A)

            # Choose the best neighbor to match against (left or up), whichever gives larger total overlap.
            candidates = []
            if j > 0:
                prev_V_left = eigvecs[i, j - 1]
                M_left = np.abs(prev_V_left.conj().T @ V)
                total_left = M_left.max(axis=1).sum()
                candidates.append(("left", total_left, prev_V_left))
            if i > 0:
                prev_V_up = eigvecs[i - 1, j]
                M_up = np.abs(prev_V_up.conj().T @ V)
                total_up = M_up.max(axis=1).sum()
                candidates.append(("up", total_up, prev_V_up))

            if candidates:
                # pick the neighbor with the highest aggregate overlap
                _, _, prev_V = max(candidates, key=lambda t: t[1])
            else:
                # only happens at (0,0), already handled
                prev_V = V0

            perm, V_matched = _match_by_overlap(prev_V, V)

            eigvals[i, j] = w[perm]
            eigvecs[i, j] = V_matched

    return deltaJ_grid, Jintra_grid, eigvals, eigvecs


# --- Example usage ---
if __name__ == "__main__":
    Jintra = 14.0
    DeltaJ = 6.5
    Energies = CalcEigenEnergies(Jintra, DeltaJ)

    # Print a quick sanity check
    print("Selected energies:", Energies)
