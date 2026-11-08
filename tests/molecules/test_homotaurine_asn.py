"""Homotaurine-specific physics checks plus reloads of both five-seed experiments."""

import json
import unittest

import numpy as np
import sympy as sp

from tests.molecules import test_dss_asn as shared
from experiments.molecules.train import (
    HOMOTAURINE as P,
    physical_bounds,
    gap_predictions,
    numerical_dataset,
)
from nmrkan.physics.real_molecules import (
    analytical_eigenvalues,
    analytical_frequencies,
    cubic_invariants,
    cross_section_taylor,
)
from nmrkan.physics.three_ch2 import hamiltonian, normalize_parameters
from experiments.molecules.paths import REFERENCE_ROOT
from nmrkan.physics.three_ch2 import EXAMPLES

OUT = REFERENCE_ROOT / "homotaurine"
Q = list(EXAMPLES.values())[1]["dJAX"]


class HomotaurineTrainedArtifactTests(shared.TrainedArtifactTests):
    folder_name = "homotaurine"

    def test_molecule_reference_splits_and_domains(self):
        for domain in ("local", "broad"):
            folder = OUT / domain
            config = json.loads((folder / "config.json").read_text())
            self.assertEqual(config["molecule"], "homotaurine")
            np.testing.assert_array_equal(config["reference_parameters_hz"], P)
            np.testing.assert_array_equal(
                config["bounds_hz"], physical_bounds(domain, P)
            )
            data = np.load(folder / "dataset.npz")
            parts = [
                data[k] for k in ("train_indices", "validation_indices", "test_indices")
            ]
            np.testing.assert_array_equal(
                np.sort(np.concatenate(parts)), np.arange(100000)
            )
            self.assertEqual([len(x) for x in parts], [80000, 10000, 10000])
            for p, y, s, r in zip(
                data["parameters_hz"][:100],
                data["targets"][:100],
                data["scale_hz"][:100],
                data["reference_hz"][:100],
            ):
                np.testing.assert_allclose(
                    r + s * y, analytical_eigenvalues(p), atol=1e-11, rtol=0
                )
        # Same broad molecular Hamiltonian benchmark as DSS, not independent data.
        old = np.load(shared.ROOT / "dss/broad/dataset.npz")
        new = np.load(OUT / "broad/dataset.npz")
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
        self.assertGreater(
            physical_bounds("local", P)[0, 1], physical_bounds("broad", P)[0, 1]
        )

    def test_section_is_substitution_into_actual_four_input_model(self):
        t = np.linspace(0, 1.5, 601)
        p = np.tile(P, (len(t), 1))
        p[:, 3:] *= t[:, None]
        data = numerical_dataset(p)
        for domain in ("local", "broad"):
            f = json.loads((OUT / domain / "learned_formulas.json").read_text())
            fn = sp.lambdify(
                sp.symbols("xA xX u v"),
                [sp.sympify(e) for e in f["formulas"]],
                "numpy",
                cse=True,
            )
            energy = np.asarray(fn(*data["features"].T)).T
            gaps = gap_predictions(energy, data["scale_hz"])
            section = np.polynomial.polynomial.polyval(
                t, np.array(f["cross_section_coefficients_hz"]).T
            ).T
            np.testing.assert_allclose(gaps, section, atol=1e-9, rtol=0)
            np.testing.assert_allclose(
                section[:, 2], section[:, 0] + section[:, 1], atol=1e-10, rtol=0
            )


class HomotaurinePhysicsTests(unittest.TestCase):
    def test_reference_section_and_cubic_coefficients(self):
        result = json.loads((OUT / "analytical_results.json").read_text())
        coefficients = result["cubic_section_coefficients_even_powers"]
        x, s, r = normalize_parameters(*P)
        previous = None
        for t in np.linspace(-1.5, 1.5, 121):
            p = P.copy()
            p[3:] *= t
            E = analytical_eigenvalues(p)
            np.testing.assert_allclose(
                E, np.linalg.eigvalsh(hamiltonian(*p)), atol=1e-11, rtol=0
            )
            np.testing.assert_allclose(
                cubic_invariants(p),
                [np.polynomial.polynomial.polyval(t * t, c) for c in coefficients],
                atol=1e-14,
            )
            self.assertTrue(np.all(np.diff(E) >= 0))
            if previous is not None:
                bound = np.linalg.norm(hamiltonian(0, 0, 0, *P[3:]), 2) * 0.025
                self.assertLessEqual(max(abs(E - previous)), bound + 1e-10)
            previous = E
        np.testing.assert_allclose(
            result["frequencies_hz"], analytical_frequencies(P), atol=1e-12
        )
        np.testing.assert_allclose(
            result["taylor_about_t1_hz"], cross_section_taylor(P), atol=1e-12
        )

    def test_optional_q_correction_and_model_errors(self):
        result = json.loads((OUT / "analytical_results.json").read_text())

        def gaps(q):
            return np.diff(np.linalg.eigvalsh(hamiltonian(*P, dJAX=q))[1:])

        eps = 1e-4
        derivative = (gaps(eps) - gaps(-eps)) / (2 * eps)
        np.testing.assert_allclose(
            result["q_gap_derivative"][:2], derivative, atol=1e-9
        )
        self.assertLess(result["q_first_order_max_error_hz"], 1e-4)
        self.assertGreater(result["parity_max_splitting_hz"], 0.036)
        self.assertLess(
            result["approximation_max_errors_hz"]["Second-order effective 3x3"], 0.0003
        )
        self.assertGreater(
            result["approximation_max_errors_hz"]["Drop type-II (bare 3x3)"], 0.05
        )


if __name__ == "__main__":
    unittest.main()
