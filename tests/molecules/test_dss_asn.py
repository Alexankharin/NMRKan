"""Checks for the actual trained ASN comparison and leakage-free data pipeline."""

import json
import unittest

import numpy as np
import sympy as sp
import torch

from nmrkan.physics.three_ch2 import hamiltonian, normalize_parameters
from experiments.molecules.train import (
    make_data,
    gap_predictions,
    metrics,
    model_predict,
    KharKAN,
)

from experiments.molecules.paths import REFERENCE_ROOT as ROOT


class DataTests(unittest.TestCase):
    def test_numerical_training_labels_match_supplied_reference(self):
        data = make_data("local", 100)
        for p, x, y, s, r in zip(
            data["parameters_hz"],
            data["features"],
            data["targets"],
            data["scale_hz"],
            data["reference_hz"],
        ):
            xx, ss, rr = normalize_parameters(*p)
            np.testing.assert_allclose(x, xx, atol=1e-14)
            np.testing.assert_allclose(
                r + s * y, np.linalg.eigvalsh(hamiltonian(*p)), atol=3e-13
            )
            self.assertTrue(np.all(np.diff(y) >= 0))

    def test_splits_are_disjoint_and_complete(self):
        data = make_data("broad", 1000)
        parts = [
            data[k] for k in ("train_indices", "validation_indices", "test_indices")
        ]
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(1000))
        self.assertEqual([len(p) for p in parts], [800, 100, 100])

    def test_gap_predictions_do_not_hide_branch_order_violations(self):
        y = np.array([[-2, 3, 1, 4.0]])
        f = gap_predictions(y, np.array([2.0]))
        np.testing.assert_array_equal(f, [[-4, 6, 2]])
        self.assertEqual(f[0, 0] + f[0, 1], f[0, 2])


class TrainedArtifactTests(unittest.TestCase):
    folder_name = "dss"

    def test_both_trained_domains_and_extracted_formulas(self):
        torch.set_num_threads(1)
        for domain in ("local", "broad"):
            folder = ROOT / self.folder_name / domain
            result = json.loads((folder / "comparison.json").read_text())
            seed = result["selected_seed"]
            self.assertEqual(
                seed,
                min(result["seeds"], key=lambda s: s["validation_rmse_dimensionless"])[
                    "seed"
                ],
            )
            state = torch.load(
                folder / f"seed_{seed}.pt", map_location="cpu", weights_only=True
            )
            model = KharKAN(tuple(state["architecture"]), include_abs=False).double()
            model.switch_to_sparsemax()
            model.load_state_dict(state["state_dict"])
            conditioning = tuple(
                np.array(state[k])
                for k in ("input_mean", "input_std", "output_mean", "output_std")
            )
            data = np.load(folder / "dataset.npz")
            x = data["features"][data["test_indices"][:200]]
            pred = model_predict(model, x, conditioning)
            formulas = json.loads((folder / "learned_formulas.json").read_text())
            fn = sp.lambdify(
                sp.symbols("xA xX u v"),
                [sp.sympify(e) for e in formulas["formulas"]],
                "numpy",
                cse=True,
            )
            np.testing.assert_allclose(np.asarray(fn(*x.T)).T, pred, atol=1e-8, rtol=0)
            arrays = np.load(folder / "heldout_predictions.npz")
            i = arrays["test_indices"]
            evaluation = metrics(
                arrays["asn"], arrays["exact"], data["scale_hz"][i], data["features"][i]
            )
            self.assertAlmostEqual(
                evaluation["frequency_rmse_hz"],
                result["evaluations"]["ASN"]["frequency_rmse_hz"],
                places=12,
            )
            self.assertTrue(
                all(
                    k >= int(0.9 * result["config"]["epochs"])
                    for k in [s["best_epoch"] for s in result["seeds"]]
                )
            )
            # No train or validation sample is inserted into this external section.
            self.assertEqual(len(result["seeds"]), 5)


if __name__ == "__main__":
    unittest.main()
