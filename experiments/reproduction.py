"""Reload completed paper runs and compare like-for-like manuscript metrics."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from experiments.paper import ROOT, basis, exact_three, grid_three, metrics, run
from nmrkan.models import KharKAN

RUN_NAMES = (
    "three_site",
    "three_site_verification",
    "four_site",
    "four_site_verification",
)
DEFAULT_RUN_ROOT = ROOT / "results/reproduced/2026-11-08/paper"


def paper_run_root():
    return Path(os.environ.get("NMRKAN_PAPER_RUNS", DEFAULT_RUN_ROOT)).resolve()


def load_run(folder):
    """Validate checkpoint/formula agreement and recompute metrics after reloading."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    if not config.get("trained") or config.get("smoke_test"):
        raise ValueError(f"{folder} is not a full training run.")
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    model = KharKAN(tuple(state["architecture"]), include_abs=False).double()
    model.switch_to_sparsemax()
    model.load_state_dict(state["state_dict"])
    model.eval()
    with np.load(folder / "dataset.npz") as data:
        x, y = data["inputs"], data["targets"]
        tr, va = data["train_indices"], data["validation_indices"]
    np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(len(x)))
    assert len(x) == config["samples"]
    with (folder / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == config["epochs"] == int(history[-1]["epoch"])
    with torch.no_grad():
        predictions = model(torch.as_tensor(x, dtype=torch.float64)).numpy()
    expressions = json.loads((folder / "learned_formulas.json").read_text())
    symbols = sp.symbols(f"x_0:{x.shape[1]}")

    def evaluate(formulas, values):
        function = sp.lambdify(
            symbols, [sp.sympify(e) for e in formulas], "numpy", cse=True
        )
        return np.column_stack(
            [np.broadcast_to(v, len(values)) for v in function(*values.T)]
        )

    symbolic = evaluate(expressions["raw"], x[va])
    np.testing.assert_allclose(symbolic, predictions[va], atol=1e-9, rtol=1e-9)
    validation = metrics(predictions[va], y[va])
    np.testing.assert_allclose(
        validation["rmse"], expressions["validation_neural"]["rmse"], atol=1e-14, rtol=0
    )
    with np.load(folder / "ols.npz") as ols:
        baseline = metrics(basis(x[va]) @ ols["train_coefficients"], y[va])
    result = {
        "config": config,
        "all_samples_neural": metrics(predictions, y),
        "validation_neural": validation,
        "validation_processed": metrics(
            evaluate(expressions["processed"], x[va]), y[va]
        ),
        "validation_ols": baseline,
        "formula_reload_max_abs_error": float(np.max(abs(symbolic - predictions[va]))),
        "support_sizes": expressions["support_sizes"],
        "raw_degrees": [
            int(sp.Poly(sp.sympify(e), *symbols).total_degree())
            for e in expressions["raw"]
        ],
        "source_files_sha256": {
            name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "model.pt",
                "dataset.npz",
                "learned_formulas.json",
                "history.csv",
            )
        },
    }
    if config["problem"] == "three_site":
        grid = grid_three()
        with torch.no_grad():
            grid_prediction = model(torch.as_tensor(grid, dtype=torch.float64)).numpy()
        result["grid_neural"] = metrics(grid_prediction, exact_three(grid))
        result["grid_processed"] = metrics(
            evaluate(expressions["processed"], grid), exact_three(grid)
        )
    return result


def compare_runs(root):
    torch.set_num_threads(1)
    runs = {name: load_run(Path(root) / name) for name in RUN_NAMES}
    # Quoted values and their last displayed decimal places in manuscript v24.
    specifications = [
        (
            "Three-site ASN, complete sample",
            "three_site",
            "all_samples_neural",
            0.0179,
            0.0001,
        ),
        (
            "Four-site primary ASN, validation",
            "four_site",
            "validation_neural",
            0.0079,
            0.0001,
        ),
        (
            "Four-site verification ASN, validation",
            "four_site_verification",
            "validation_neural",
            0.00753,
            0.00001,
        ),
        (
            "Four-site verification OLS, validation",
            "four_site_verification",
            "validation_ols",
            0.00702,
            0.00001,
        ),
    ]
    comparison = []
    for label, name, key, reported, precision in specifications:
        actual = runs[name][key]["rmse"]
        comparison.append(
            {
                "metric": label,
                "run": name,
                "reported_rmse": reported,
                "rerun_rmse": actual,
                "relative_difference_percent": 100 * (actual / reported - 1),
                "matches_displayed_precision": abs(actual - reported) < precision / 2,
            }
        )
    return {
        "runs": runs,
        "comparison": comparison,
        "interpretation": "Fresh CPU runs with recorded seed 42. Original CUDA model initialization seeds/checkpoints were not supplied. The uniform-ratio verification does not reproduce the stated manuscript accuracy; do not change sampling or select seeds to force a numerical match.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=paper_run_root())
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train all four protocols into a fresh root before comparing.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Save the reload audit and manuscript comparison as JSON.",
    )
    args = parser.parse_args()
    if args.train:
        if (
            args.root.resolve() == DEFAULT_RUN_ROOT.resolve()
            or args.root.resolve().is_relative_to((ROOT / "results").resolve())
        ):
            parser.error(
                "Train into a new directory under outputs/, not the versioned results."
            )
        if any((args.root / name).exists() for name in RUN_NAMES):
            parser.error("Choose a fresh --root; existing runs are preserved.")
        for name in RUN_NAMES:
            config = json.loads((ROOT / "configs/paper" / f"{name}.json").read_text())
            run(config, args.root / name, train=True)
    summary = compare_runs(args.root)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    for item in summary["comparison"]:
        print(
            f"{item['metric']}: paper={item['reported_rmse']:.6g}, rerun={item['rerun_rmse']:.8g}"
        )


if __name__ == "__main__":
    main()
