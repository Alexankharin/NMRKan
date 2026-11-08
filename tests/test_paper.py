"""Independent physical invariants and manuscript numerical reference values."""

import json

import numpy as np
import pytest

from experiments.paper import (
    ROOT,
    exact_three,
    exact_four,
    perturbation_three,
    grid_three,
    make_dataset,
    metrics,
    published_predictions,
    run,
)
from nmrkan.physics.three_ch2 import hamiltonian


def test_three_site_signed_energy_convention():
    x = np.random.default_rng(17).uniform(-2, 2, (100, 2))
    expected = np.array(
        [
            np.linalg.eigvalsh(hamiltonian(-14, -14, -14, -14 * a, -14 * b)) / -14
            for a, b in x
        ]
    )
    np.testing.assert_allclose(exact_three(x), expected, atol=1e-14)
    np.testing.assert_allclose(exact_three(x), exact_three(abs(x)), atol=1e-14)
    np.testing.assert_allclose(exact_three(x), exact_three(x[:, ::-1]), atol=1e-14)
    np.testing.assert_allclose(exact_three(x).sum(axis=1), -3, atol=1e-14)


def test_local_perturbation_matches_manuscript_precision():
    x = grid_three(0.2, 101)
    error = metrics(perturbation_three(x), exact_three(x))
    assert abs(error["rmse"] - 4.28e-5) < 5e-8
    assert abs(error["max_abs_error"] - 2.64e-4) < 5e-7
    np.testing.assert_array_equal(
        perturbation_three([[0, 0]]), [[0.75, -1.25, -1.25, -1.25]]
    )


def test_published_three_site_formulas_on_independent_grid():
    x = grid_three()
    assert (
        abs(metrics(published_predictions(x), exact_three(x))["rmse"] - 0.0190) < 0.0001
    )


def test_four_site_reflection_and_zero_coupling():
    r = np.linspace(0, 3, 101)[:, None]
    y = exact_four(r)
    np.testing.assert_allclose(y + y[:, ::-1], -2, atol=2e-14)
    np.testing.assert_array_equal(y[0], [0, 0, 0, 0, -2, -2, -2, -2])
    np.testing.assert_allclose(y, exact_four(-r), atol=1e-14)


def test_four_site_against_full_spin_basis():
    # Eq. (8): each nearest-neighbour pair flips with matrix element deltaJ/2.
    # Project an independently built 16x16 spin operator onto odd parity.
    from data_generation.eigenvalues_4CH2 import Ham4CH2s3Sand1S

    states = [8, 4, 2, 1, 7, 11, 13, 14]
    for j, delta in [(-14.0, 3.0), (1.0, 2.7)]:
        full = np.diag([j * (state.bit_count() - 3) for state in range(16)])
        for state in range(16):
            for bond in range(3):
                full[state, state ^ (3 << bond)] = delta / 2
        np.testing.assert_array_equal(
            full[np.ix_(states, states)], Ham4CH2s3Sand1S(j, delta)
        )


@pytest.mark.parametrize("name", ["three_site", "four_site_verification"])
def test_saved_split_and_smoke_training(tmp_path, name):
    config = json.loads((ROOT / f"configs/paper/{name}.json").read_text())
    config.update(samples=128, epochs=10)
    x, y, tr, va = make_dataset(config)
    assert set(tr).isdisjoint(va)
    np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(128))
    result = run(config, tmp_path, train=True)
    assert np.isfinite(result["fresh_asn"]["validation_neural"]["rmse"])
    assert result["fresh_asn"]["extraction_max_abs_difference"] < 1e-9
    assert (tmp_path / "model.pt").is_file()
    figure = "fig04_surfaces.pdf" if name == "three_site" else "fig05_branches.pdf"
    assert (tmp_path / figure).is_file()
