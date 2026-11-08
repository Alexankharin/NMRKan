"""Recompute manuscript benchmarks and optionally train an ASN from scratch.

Run from the checkout: python -m experiments.paper --help.
Published rounded expressions and fresh neural fits are evaluated separately.
"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from data_generation.eigenvalues_4CH2 import Ham4CH2s3Sand1S
from data_generation.nmr_datagen import ham_reduced_symmetric

ROOT = Path(__file__).resolve().parents[1]
# Columns: 1, r1+r2, r1**2+r2**2, r1*r2. Manuscript Eq. (12).
THREE_COEFFICIENTS = np.array(
    [
        [0.743, 0.022, 0.094, 0.023],
        [-1.233, 0.361, 0.104, -0.372],
        [-1.239, -0.028, -0.133, 0.288],
        [-1.272, -0.355, -0.062, 0.054],
    ]
)
# Columns: 1, r, r**2. Manuscript Eq. (13).
FOUR_COEFFICIENTS = np.array(
    [
        [-0.016, 0.915, 0.128],
        [-0.006, 0.351, 0.032],
        [0, -0.337, 0.188],
        [-0.010, -0.772, 0.282],
        [-1.991, 0.775, -0.284],
        [-2.001, 0.338, -0.188],
        [-1.995, -0.349, -0.034],
        [-1.979, -0.933, -0.118],
    ]
)


def exact_three(x):
    """Descending E/J, equivalent to diagonalizing H at signed J < 0."""
    x = np.asarray(x, dtype=float).reshape(-1, 2)
    base = ham_reduced_symmetric(1, 1, 1, 0, 0, 0)
    a = ham_reduced_symmetric(0, 0, 0, 1, 0, 0)
    b = ham_reduced_symmetric(0, 0, 0, 0, 1, 0)
    return np.linalg.eigvalsh(base + x[:, 0, None, None] * a + x[:, 1, None, None] * b)[
        :, ::-1
    ].copy()


def exact_four(x):
    r = np.asarray(x, dtype=float).reshape(-1)
    h = Ham4CH2s3Sand1S(1, 0) + r[:, None, None] * Ham4CH2s3Sand1S(0, 1)
    return np.linalg.eigvalsh(h)[:, ::-1].copy()


def perturbation_three(x):
    """Eq. (10) in dimensionless descending order, with the limit at S=0."""
    a, b = np.asarray(x, dtype=float).reshape(-1, 2).T
    s = a * a + b * b
    mixed = np.divide(a * a * b * b, s, out=np.zeros_like(s), where=s != 0)
    difference = np.divide((b * b - a * a) ** 2, s, out=np.zeros_like(s), where=s != 0)
    return np.column_stack(
        (
            0.75 + s / 8,
            -1.25 + np.sqrt(s) / 2 - mixed / 4,
            -1.25 - difference / 8,
            -1.25 - np.sqrt(s) / 2 - mixed / 4,
        )
    )


def basis(x, radial=False):
    x = np.asarray(x).reshape(len(x), -1)
    if x.shape[1] == 1:
        r = x[:, 0]
        return np.column_stack((np.ones(len(r)), r, r * r))
    a, b = x.T
    columns = [np.ones(len(a)), a + b, a * a + b * b, a * b]
    if radial:
        columns.append(np.hypot(a, b))
    return np.column_stack(columns)


def published_predictions(x):
    coefficients = (
        THREE_COEFFICIENTS if np.asarray(x).shape[1] == 2 else FOUR_COEFFICIENTS
    )
    return basis(x) @ coefficients.T


def metrics(prediction, target):
    residual = np.asarray(prediction) - target
    ss = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    # R² is undefined on a constant branch (e.g. the uniform-chain epsilon_3).
    nonconstant = ss > 1e-24 * len(target) * np.maximum(
        1, np.max(abs(target), axis=0) ** 2
    )
    r2 = 1 - np.divide(
        np.sum(residual**2, axis=0), ss, out=np.full_like(ss, np.nan), where=nonconstant
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error": float(np.max(abs(residual))),
        "r2_per_branch": [float(value) if np.isfinite(value) else None for value in r2],
    }


def grid_three(limit=2, points=121):
    a, b = np.meshgrid(np.linspace(0, limit, points), np.linspace(0, limit, points))
    return np.column_stack((a.ravel(), b.ravel()))


def make_dataset(config):
    rng = np.random.RandomState(config["data_seed"])
    n = config["samples"]
    if config["problem"] == "three_site":
        # Match the original notebook's separate draws for the two coordinates.
        x = np.column_stack((rng.uniform(0, 2, n), rng.uniform(0, 2, n)))
        y = exact_three(x)
    elif config["sampling"] == "interpolated_physical":
        from data_generation.datasets import make_normalized_eigenvalue_dataset

        np.random.seed(config["data_seed"])
        data = make_normalized_eigenvalue_dataset(
            n, grid_resolution=config["grid_resolution"]
        )
        x = data["train_input"].numpy().astype(float)
        y = np.sort(data["train_label"].numpy().astype(float), axis=1)[:, ::-1].copy()
    else:
        x = rng.uniform(0, 3, (n, 1))
        y = exact_four(x)
    order = torch.randperm(
        n, generator=torch.Generator().manual_seed(config["split_seed"])
    ).numpy()
    return x, y, order[: int(0.8 * n)], order[int(0.8 * n) :]


def train_asn(config, x, y, train, validation, out):
    """A fresh run of the stated protocol, not a recovered historical checkpoint."""
    import sympy as sp

    from nmrkan.models import KharKAN

    torch.manual_seed(config["model_seed"])
    torch.use_deterministic_algorithms(True)
    model = KharKAN(tuple(config["architecture"]), include_abs=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    inputs = torch.tensor(x, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)
    history = []
    switch = int(config["softmax_fraction"] * config["epochs"])
    for epoch in range(config["epochs"]):
        if epoch == switch:
            model.switch_to_sparsemax()
        indices = torch.tensor(train)[torch.randperm(len(train))]
        total = 0.0
        for batch in indices.split(config["batch_size"]):
            optimizer.zero_grad()
            error = model(inputs[batch]) - targets[batch]
            fit = (
                error.abs().mean() if config["loss"] == "mae" else error.square().mean()
            )
            penalty = (
                config["regularization"] * model.L05_loss() if epoch < switch else 0
            )
            (fit + penalty).backward()
            optimizer.step()
            total += float(fit.detach()) * len(batch)
        with torch.no_grad():
            score = metrics(model(inputs[validation]).numpy(), y[validation])["rmse"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / len(train),
                "validation_rmse": score,
                "sparsemax": epoch >= switch,
            }
        )
        if epoch % 100 == 0 or epoch == config["epochs"] - 1:
            print(
                f"Epoch {epoch + 1}/{config['epochs']}: validation RMSE {score:.6g}",
                flush=True,
            )
    # Final epoch is used; no historical selection rule is invented.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": config["architecture"],
            "sparsemax": True,
            "config": config,
        },
        out / "model.pt",
    )
    with (out / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)
    model.double()
    raw = list(model.symbolic_formula(clean=False, round_digits=15).values())
    symbols = sp.symbols(f"x_0:{x.shape[1]}")
    processed = []
    for expression in raw:
        if len(symbols) == 2:
            a, b = symbols
            expression = sp.expand((expression + expression.xreplace({a: b, b: a})) / 2)
        processed.append(
            sp.Add(
                *[
                    sp.Float(round(float(coefficient), 3))
                    * sp.prod(s**p for s, p in zip(symbols, power))
                    for power, coefficient in sp.Poly(expression, *symbols).terms()
                    if abs(float(coefficient)) >= 1e-3
                ]
            )
        )

    def predict(values):
        with torch.no_grad():
            return model(torch.as_tensor(values, dtype=torch.float64)).numpy()

    def symbolic_values(expressions, values):
        fn = sp.lambdify(symbols, expressions, "numpy", cse=True)
        return np.column_stack([np.broadcast_to(v, len(values)) for v in fn(*values.T)])

    neural = predict(x[validation])
    extracted = symbolic_values(raw, x[validation])
    np.testing.assert_allclose(extracted, neural, atol=1e-9, rtol=1e-9)
    result = {
        "raw": [str(e) for e in raw],
        "processed": [str(e) for e in processed],
        "validation_neural": metrics(neural, y[validation]),
        "validation_processed": metrics(
            symbolic_values(processed, x[validation]), y[validation]
        ),
        "extraction_max_abs_difference": float(np.max(abs(extracted - neural))),
        "support_sizes": [
            (layer._activation_weights().detach().numpy() != 0).sum(axis=1).tolist()
            for layer in model.layers
        ],
    }
    (out / "learned_formulas.json").write_text(json.dumps(result, indent=2) + "\n")
    return predict, result


def figures(problem, out, predictor=None):
    import matplotlib.pyplot as plt

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if problem == "three_site":
        axis = np.linspace(-2, 2, 121)
        a, b = np.meshgrid(axis, axis)
        x = np.column_stack((a.ravel(), b.ravel()))
        exact = exact_three(x)
        predicted = (
            published_predictions(abs(x)) if predictor is None else predictor(abs(x))
        )
        label = "Eq. (12), rounded" if predictor is None else "Fresh ASN"
        fig, axes = plt.subplots(4, 2, figsize=(8, 11), layout="constrained")
        for k in range(4):
            low, high = (
                min(exact[:, k].min(), predicted[:, k].min()),
                max(exact[:, k].max(), predicted[:, k].max()),
            )
            for j, (values, name) in enumerate([(exact, "Exact"), (predicted, label)]):
                im = axes[k, j].imshow(
                    values[:, k].reshape(a.shape),
                    origin="lower",
                    extent=(-2, 2, -2, 2),
                    vmin=low,
                    vmax=high,
                    aspect="auto",
                )
                axes[k, j].set(
                    title=f"{name}, branch {k + 1}", xlabel="r1", ylabel="r2"
                )
                fig.colorbar(im, ax=axes[k, j])
        save(fig, "fig04_surfaces")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, values, title in [
            (axes[0], np.column_stack((axis, axis)), "Uniform chain"),
            (axes[1], np.column_stack((axis, np.ones_like(axis))), "r2 = 1"),
        ]:
            ax.plot(axis, exact_three(values))
            if ax is axes[0]:
                # Signed perturbative branches cross at zero, as in Fig. 3(b).
                signed = np.column_stack(
                    (
                        0.75 + axis**2 / 4,
                        -1.25 + axis / np.sqrt(2) - axis**2 / 8,
                        np.full_like(axis, -1.25),
                        -1.25 - axis / np.sqrt(2) - axis**2 / 8,
                    )
                )
                ax.plot(axis, signed, "k--", alpha=0.45)
            ax.set(title=title, xlabel="r1", ylabel="E/J")
        save(fig, "fig03_sections")
        np.savez_compressed(
            out / "figure_data.npz", inputs=x, exact=exact, predicted=predicted
        )
    else:
        x = np.linspace(0, 3, 601)[:, None]
        exact = exact_four(x)
        predicted = published_predictions(x) if predictor is None else predictor(x)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(x, exact, color="steelblue")
        ax.plot(x, predicted, "--", color="firebrick")
        ax.plot([], [], color="steelblue", label="Exact")
        ax.plot(
            [],
            [],
            "--",
            color="firebrick",
            label="Eq. (13), rounded" if predictor is None else "Fresh ASN",
        )
        ax.set(xlabel="r", ylabel="E/J")
        ax.legend()
        save(fig, "fig05_branches")
        np.savez_compressed(
            out / "figure_data.npz", inputs=x, exact=exact, predicted=predicted
        )


def run(config, out, train=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(f"{out} already contains a run; choose a new --output.")
    torch.set_num_threads(1)
    config = dict(config)
    config.update(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        trained=train,
        units="dimensionless E/J; descending branch order",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    x, y, tr, va = make_dataset(config)
    np.savez_compressed(
        out / "dataset.npz",
        inputs=x,
        targets=y,
        train_indices=tr,
        validation_indices=va,
    )
    design = basis(x)
    coeff = np.linalg.lstsq(design[tr], y[tr], rcond=None)[0]
    full_coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    result = {
        "ols_validation": metrics(design[va] @ coeff, y[va]),
        "ols_full_sample_fit": metrics(design @ full_coeff, y),
        "published_rounded_validation": metrics(published_predictions(x[va]), y[va]),
    }
    np.savez(
        out / "ols.npz", train_coefficients=coeff, full_sample_coefficients=full_coeff
    )
    if config["problem"] == "three_site":
        local = grid_three(0.2, 101)
        local_y = exact_three(local)
        global_grid = grid_three()
        uniform = np.repeat(np.linspace(0, 2, 1001)[:, None], 2, axis=1)
        radial = basis(x, radial=True)
        radial_coeff = np.linalg.lstsq(radial[tr], y[tr], rcond=None)[0]
        np.save(out / "radial_coefficients.npy", radial_coeff)
        result.update(
            radial_validation=metrics(radial[va] @ radial_coeff, y[va]),
            radial_local_grid=metrics(basis(local, True) @ radial_coeff, local_y),
            ols_local_grid=metrics(basis(local) @ coeff, local_y),
            perturbation_local_grid=metrics(perturbation_three(local), local_y),
            published_rounded_grid=metrics(
                published_predictions(global_grid), exact_three(global_grid)
            ),
            published_rounded_uniform=metrics(
                published_predictions(uniform), exact_three(uniform)
            ),
            perturbation_uniform=metrics(
                perturbation_three(uniform), exact_three(uniform)
            ),
        )
    predictor = None
    if train:
        predictor, result["fresh_asn"] = train_asn(config, x, y, tr, va, out)
    figures(config["problem"], out, predictor)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/paper/three_site.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run neural optimization; otherwise only exact/rounded/OLS calculations.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="256 samples and 10 epochs for a workflow check, not paper metrics.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.quick:
        config.update(samples=256, epochs=10, grid_resolution=20, smoke_test=True)
    out = args.output or ROOT / "outputs/paper" / (
        args.config.stem
        + ("_quick" if args.quick else "")
        + ("_trained" if args.train else "")
    )
    if (
        ROOT / "results/reference"
    ).resolve() in out.resolve().parents or out.resolve() == (
        ROOT / "results/reference"
    ).resolve():
        parser.error("Choose an output directory outside results/reference.")
    run(config, out, args.train)


if __name__ == "__main__":
    main()
