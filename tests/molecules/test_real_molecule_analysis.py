"""Independent numerical and symbolic validation of the DSS analytical solution.

Run: python -m pytest tests/molecules/test_real_molecule_analysis.py
SymPy is optional; its absence skips only the exact polynomial identities.
"""

import unittest

import numpy as np

from nmrkan.physics.three_ch2 import (
    dimensionless_matrix,
    hamiltonian,
    normalize_parameters,
    self_check,
)
from nmrkan.physics.real_molecules import (
    analytical_eigenvalues,
    analytical_frequencies,
    cross_section_taylor,
    effective_three_state,
    mode_matrix,
    normal_modes,
    principal_gaps,
    singlet_observable,
    spectral_lines,
)


class AnalyticalTests(unittest.TestCase):
    def test_independent_six_spin_projection(self):
        self.assertTrue(self_check().startswith("PASS"))

    def test_2000_random_parameter_points(self):
        rng = np.random.default_rng(88124)
        for _ in range(2000):
            p = np.r_[rng.uniform(-15, -12.5, 3), rng.uniform(-7, 7, 2)]
            exact = np.linalg.eigvalsh(hamiltonian(*p))
            np.testing.assert_allclose(
                analytical_eigenvalues(p), exact, atol=3e-11, rtol=0
            )
            np.testing.assert_allclose(
                analytical_frequencies(p), principal_gaps(exact[1:]), atol=3e-11, rtol=0
            )
            np.testing.assert_allclose(
                normal_modes(p), normal_modes(p, "svd"), atol=3e-11, rtol=0
            )

    def test_degenerate_and_decoupled_limits(self):
        for a, b, c in [
            (-14, -14, -14),
            (-13, -14, -15),
            (-13, -13, -15),
            (-15, -13, -13),
        ]:
            for d, e in [(0, 0), (0, 6), (6, 0), (1e-10, 1e-10), (1e-7, 0)]:
                p = (a, b, c, d, e)
                np.testing.assert_allclose(
                    analytical_eigenvalues(p),
                    np.linalg.eigvalsh(hamiltonian(*p)),
                    atol=3e-9,
                    rtol=0,
                )

    def test_normalization_and_energy_reference(self):
        p = np.array([-13.758727, -13.408159, -14.121958, 5.269170, 6.597217])
        x, s, r = normalize_parameters(*p)
        np.testing.assert_allclose(
            analytical_eigenvalues(p),
            r + s * np.linalg.eigvalsh(dimensionless_matrix(*x)),
            atol=3e-12,
        )
        for factor in [0.01, 3.7, 100]:
            np.testing.assert_allclose(
                analytical_eigenvalues(factor * p),
                factor * analytical_eigenvalues(p),
                atol=1e-10,
            )
            np.testing.assert_allclose(
                mode_matrix(factor * p), mode_matrix(p), atol=2e-15
            )

    def test_sorted_cusp_without_jumps_at_crossing(self):
        ts = np.linspace(-1, 1, 1001)
        E = np.array(
            [analytical_eigenvalues((-14, -14, -14, 6 * t, 6 * t)) for t in ts]
        )
        self.assertTrue(np.all(np.diff(E, axis=1) >= -1e-12))
        np.testing.assert_allclose(E, E[::-1], atol=1e-11)
        # Weyl continuity bound: ||dH/dt||_2 * dt bounds each ordered branch step.
        direction = hamiltonian(0, 0, 0, 6, 6)
        bound = np.linalg.norm(direction, 2) * (ts[1] - ts[0])
        self.assertLessEqual(np.max(abs(np.diff(E, axis=0))), bound + 1e-11)

    def test_gap_identities_and_parity(self):
        p = (-13.758727, -13.408159, -14.121958, 5.269170, 6.597217)
        f = analytical_frequencies(p)
        self.assertAlmostEqual(f[0] + f[1], f[2], places=13)
        odd = np.linalg.eigvalsh(hamiltonian(*p, parity="odd"))
        np.testing.assert_allclose(principal_gaps(odd[:3])[[1, 0, 2]], f, atol=3e-13)
        even_q = np.linalg.eigvalsh(hamiltonian(*p, dJAX=0.00249))
        odd_q = np.linalg.eigvalsh(hamiltonian(*p, dJAX=0.00249, parity="odd"))
        self.assertGreater(
            np.max(
                abs(principal_gaps(even_q[1:]) - principal_gaps(odd_q[:3])[[1, 0, 2]])
            ),
            0.001,
        )

    def test_spectrum_matches_direct_complex_propagation(self):
        rng = np.random.default_rng(64)
        H = hamiltonian(-13, -14, -15, 5, 6)
        matrices = []
        for _ in range(2):
            m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            matrices.append(m + m.conj().T)
        rho, O = matrices
        dc, lines = spectral_lines(H, rho, O)
        E, V = np.linalg.eigh(H)
        for t in [0, 0.001, 0.3, 1.7]:
            U = (V * np.exp(-2j * np.pi * E * t)) @ V.conj().T
            direct = np.trace(O @ U @ rho @ U.conj().T).real
            reconstructed = dc + sum(
                r["cosine"] * np.cos(2 * np.pi * r["frequency_hz"] * t)
                + r["sine"] * np.sin(2 * np.pi * r["frequency_hz"] * t)
                for r in lines
            )
            self.assertAlmostEqual(direct, reconstructed, places=10)

    def test_autocorrelation_sum_and_stationary_limit(self):
        O = singlet_observable("A")
        for p in [(-13, -14, -15, 5, 6), (-14, -14, -14, 0, 0)]:
            dc, lines = spectral_lines(hamiltonian(*p), O, O)
            self.assertAlmostEqual(
                dc + sum(r["cosine"] for r in lines), np.trace(O @ O), places=13
            )
            self.assertTrue(all(r["cosine"] >= -1e-14 for r in lines))
        self.assertTrue(all(abs(r["cosine"]) < 1e-14 for r in lines))

    def test_effective_three_state_accuracy_at_dss(self):
        p = (-13.758727, -13.408159, -14.121958, 5.269170, 6.597217)
        f = analytical_frequencies(p)
        bare = principal_gaps(np.linalg.eigvalsh(hamiltonian(*p)[:3, :3]))
        corrected = principal_gaps(np.linalg.eigvalsh(effective_three_state(p)))
        self.assertGreater(max(abs(bare - f)), 0.3)
        self.assertLess(max(abs(corrected - f)), 0.009)

    def test_analytic_cross_section_derivatives(self):
        p = np.array([-13.758727, -13.408159, -14.121958, 5.269170, 6.597217])
        coefficients = cross_section_taylor(p)
        h = 1e-3

        def f(t):
            pt = p.copy()
            pt[3:] *= t
            return analytical_frequencies(pt)

        np.testing.assert_allclose(
            coefficients[:, 1], (f(1 + h) - f(1 - h)) / (2 * h), atol=2e-7, rtol=0
        )
        np.testing.assert_allclose(
            coefficients[:, 2],
            (f(1 + h) + f(1 - h) - 2 * f(1)) / (2 * h * h),
            atol=2e-7,
            rtol=0,
        )

    def test_exact_symbolic_polynomial_identities(self):
        try:
            import sympy as sp
        except ImportError:
            self.skipTest("SymPy not installed")
        a, b, c, d, e, z = sp.symbols("a b c d e z")
        total = a + b + c
        G = sp.diag(a - total / 2, b - total / 2, c - total / 2, total / 2)
        for i, j, v in [(0, 1, d), (1, 2, e), (0, 3, e), (2, 3, d)]:
            G[i, j] = G[j, i] = v / 2
        c1 = a * a + b * b + c * c + d * d + e * e
        c2 = (
            a * a * b * b
            + a * a * c * c
            + b * b * c * c
            + a * a * e * e
            + c * c * d * d
            + d * d * e * e
        )
        c3 = (a * b * c) ** 2
        self.assertEqual(
            sp.expand(
                G.charpoly(z).as_expr()
                - (z**4 - c1 * z * z / 2 - a * b * c * z + (c1 * c1 - 4 * c2) / 16)
            ),
            0,
        )
        B = sp.diag(a, b, c)
        B[0, 1] = d
        B[1, 2] = e
        self.assertEqual(
            sp.expand(
                (B * B.T).charpoly(z).as_expr() - (z**3 - c1 * z * z + c2 * z - c3)
            ),
            0,
        )
        # Reconstruct the quartic from the four sign-combinations independently.
        u, v, w = sp.symbols("u v w", positive=True)
        sign_roots = [
            (-u - v - w) / 2,
            (u + v - w) / 2,
            (u - v + w) / 2,
            (-u + v + w) / 2,
        ]
        product = sp.prod(z - r for r in sign_roots)
        t1 = u * u + v * v + w * w
        t2 = u * u * v * v + u * u * w * w + v * v * w * w
        self.assertEqual(
            sp.expand(
                product
                - (z**4 - t1 * z * z / 2 + u * v * w * z + (t1 * t1 - 4 * t2) / 16)
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
