import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from src.datagen.data_generator import ham_reduced_symmetric, compute_transition_frequency, ham_reduced_antisymmetric, HMatrixIdealized
from scipy.linalg import eigh

def get_ham_simple(Jintra, deltaJ):
    """
    Generates the Hamiltonian matrix for a simple system with given parameters.
    
    Parameters:
    Jintra (float): Intra-molecular coupling constant.
    deltaJ (float): Energy difference between states.
    
    Returns:
    np.ndarray: Hamiltonian matrix.
    """
    JAA=JMM=JXX=Jintra
    JMA=JMX=deltaJ
    JAX=0
    matrix=ham_reduced_symmetric(JAA, JMM, JXX, JMA, JMX, JAX)
    return matrix

def get_levels(matrix):
    """
    Computes the eigenvalues and eigenvectors of the Hamiltonian matrix.
    
    Parameters:
    matrix (np.ndarray): Hamiltonian matrix.
    
    Returns:
    tuple: Eigenvalues and eigenvectors.
    """
    evals, evecs = eigh(matrix)
    levels=evals[evals>1e-8]
    levels.sort()
    # get all paired level differences using itertools
    energies=[]
    for i in range(len(levels)):
         for j in range(i+1, len(levels)):
            energies.append(levels[j]-levels[i])
    return energies

def get_frequenves(
    Jintra: np.ndarray, 
    deltaJ: np.ndarray = None
) -> list:
    """
    Computes the transition frequencies for a batch of Hamiltonian matrices.

    Parameters:
    Jintra (np.ndarray or float): Intra-molecular coupling constant(s), shape (B,) or (B,2) or scalar.
    deltaJ (np.ndarray or float, optional): Energy difference(s) between states, shape (B,) or scalar.

    Returns:
    list: List of transition frequencies for each input pair.
    """
    # Handle input: allow (B,2) or two arrays of shape (B,)
    if deltaJ is None:
        Jintra = np.asarray(Jintra)
        if Jintra.ndim == 2 and Jintra.shape[1] == 2:
            Jintra_arr = Jintra[:, 0]
            deltaJ_arr = Jintra[:, 1]
        else:
            raise ValueError("If deltaJ is None, Jintra must have shape (B,2)")
    else:
        Jintra_arr = np.asarray(Jintra)
        deltaJ_arr = np.asarray(deltaJ)
        if Jintra_arr.shape != deltaJ_arr.shape:
            raise ValueError("Jintra and deltaJ must have the same shape")
    # Now Jintra_arr and deltaJ_arr are both (B,)
    results = []
    for j, d in zip(Jintra_arr, deltaJ_arr):
        matrix = get_ham_simple(j, d)
        levels = get_levels(matrix)
        results.append(levels)
    return results