import numpy as np
from scipy.linalg import eigh

# — exactly the three probe‐states from the notebook —
state1 = np.array([ 1.0,  np.sqrt(2),  1.0, 0.0 ]) / 2.0
state2 = np.array([ np.sqrt(2), 0.0, -np.sqrt(2), 0.0 ]) / 2.0
state3 = np.array([ 1.0, -np.sqrt(2),  1.0, 0.0 ]) / 2.0

def ham_reduced_symmetric(JAA, JMM, JXX, dJAM, dJMX, dJAX):
    """
    Direct port of Mathematica‘s HamReducedSymmetric:
      Diagonals = ¼⋅{−3 JAA + JMM −3 JXX,  JAA −3 JMM + JXX,  same,  JAA + JMM −3 JXX}
      Off–diagonals = {dJAX/2, dJAM/2, JMM, dJMX/2, dJAX/2}
    """
    H = np.zeros((4,4), float)

    H[0,0] = (  -3*JAA +     JMM - 3*JXX )/4.0
    H[1,1] = (    JAA - 3*JMM +     JXX )/4.0
    H[2,2] = (    JAA - 3*JMM +     JXX )/4.0
    H[3,3] = (    JAA +     JMM - 3*JXX )/4.0

    H[0,1] = H[1,0] = dJAX/2.0
    H[0,2] = H[2,0] = dJAM/2.0
    H[1,2] = H[2,1] =       JMM
    H[1,3] = H[3,1] = dJMX/2.0
    H[2,3] = H[3,2] = dJAX/2.0

    return H

def hmatrix_idealized(Jintra, deltaJ):
    # in the notebook: JAA = JMM = JXX = Jintra, and dJAM = dJMX = deltaJ, dJAX = 0
    return ham_reduced_symmetric(Jintra, Jintra, Jintra,
                                 deltaJ, deltaJ, 0.0)

def calc_freqs_idealized(Jintra, deltaJ):
    # 1) build and diagonalize
    H  = hmatrix_idealized(Jintra, deltaJ)
    w, v = eigh(H)

    # 2) project each of the 3 special states onto all 4 eigenvectors
    SP1 = v.T @ state1
    SP2 = v.T @ state2
    SP3 = v.T @ state3

    # 3) pick out which eigenvector each state overlaps most strongly with
    i1, i2, i3 = np.argmax(np.abs(SP1)), np.argmax(np.abs(SP2)), np.argmax(np.abs(SP3))
    E1, E2, E3 = w[i1], w[i2], w[i3]

    # 4) form the three transition frequencies exactly as in Mathematica
    f1 = abs(E2 - E3)
    f2 = abs(E1 - E2)
    f3 = abs(E1 - E3)
    return f1, f2, f3

# — sanity check — should print (≈3.92072, 4.57615, 8.49687)
print(calc_freqs_idealized(-14.0, 6.06))
