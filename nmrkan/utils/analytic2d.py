import math
from typing import Tuple, List, Union

import numpy as np

def _cbrt_real(x: float) -> float:
    """Real cube root (handles negative x correctly)."""
    if x >= 0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))

import numpy as np
import cmath
from typing import Tuple, Union

def _eigs_quartic_ferrari_depressed(p: float, q: float, r: float) -> np.ndarray:
    """
    Solve x^4 + p x^2 + q x + r = 0 using Ferrari via factorization.

    Uses the resolvent cubic:
        y^3 - p y^2 - 4 r y + (4 p r - q^2) = 0
    and factors into two quadratics.

    Returns 4 complex roots (typically real for your symmetric matrix).
    """
    # Solve resolvent cubic for y (complex allowed)
    coeff = [1.0, -p, -4.0 * r, 4.0 * p * r - q * q]
    ys = np.roots(coeff)

    # Choose the y that gives the smallest residual when reconstructing roots
    best_roots = None
    best_res = None

    def poly(x):
        return x**4 + p*x**2 + q*x + r

    for y in ys:
        # m^2 = y - p
        m = cmath.sqrt(y - p)
        if abs(m) < 1e-14:
            continue

        # enforce q = m * d  => d = q/m, while also d^2 = y^2 - 4r implicitly
        d = q / m

        n = (y - d) / 2.0
        k = (y + d) / 2.0

        # (x^2 + m x + n)(x^2 - m x + k) = 0
        roots = []
        roots.extend(np.roots([1.0, m, n]))
        roots.extend(np.roots([1.0, -m, k]))

        res = max(abs(poly(rt)) for rt in roots)
        if best_res is None or res < best_res:
            best_res = res
            best_roots = roots

    return np.array(best_roots, dtype=complex)


def eigenvalues_ham_reduced_symmetric_ab(
    a: Union[float, np.ndarray],
    b: Union[float, np.ndarray],
) -> Union[Tuple[float, float, float, float], np.ndarray]:
    """
    Analytic eigenvalues of your 4x4 H for:
      JAA=JMM=JXX=1, dJAX=0, a=dJAM, b=dJMX.

    Returns sorted eigenvalues. Accepts scalar or ndarray inputs.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Shapes of a and b must match")

    # Helper for one pair
    def _one(a0: float, b0: float):
        s = a0*a0 + b0*b0
        delta = a0*a0 - b0*b0

        # depressed quartic for x where λ = x - 3/4:
        # x^4 + p x^2 + q x + r = 0
        p = -(s + 3.0) / 2.0
        q = -1.0
        r = (delta*delta) / 16.0 + s / 8.0 - 3.0 / 16.0

        x_roots = _eigs_quartic_ferrari_depressed(p, q, r)

        # λ = x - 3/4
        lam = x_roots - 0.75

        # For real symmetric matrix, imaginary parts should be ~0 numerically
        lam_real = np.real_if_close(lam, tol=1e5).astype(float)
        lam_real.sort()
        return lam_real

    if a_arr.ndim == 0:
        out = _one(float(a_arr), float(b_arr))
        return tuple(out.tolist())

    flat = np.empty((a_arr.size, 4), dtype=float)
    for idx, (aa, bb) in enumerate(zip(a_arr.flat, b_arr.flat)):
        flat[idx, :] = _one(float(aa), float(bb))
    return flat.reshape(a_arr.shape + (4,))
