"""Train the repository ASN/KharKAN for DSS or homotaurine.

This is actual activation-selection training, not a fit labeled as KAN.
Targets come only from independent numerical diagonalization of the 4x4 block.
Default: 100000 physical samples, 80/10/10 split, five initialization seeds,
(4,16,4), {identity,square,zero}, Adam/MAE, 90% softmax+0.1 L05,
10% sparsemax. Affine input/output conditioning uses training data only.

python -m experiments.molecules.train --domain local
python -m experiments.molecules.train --domain broad
python -m experiments.molecules.train --molecule homotaurine --domain local
python -m experiments.molecules.train --molecule homotaurine --domain broad
"""

import argparse
import csv
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from .paths import OUTPUT_ROOT as ROOT, PROJECT_ROOT
from nmrkan.models import KharKAN
from nmrkan.physics.three_ch2 import EXAMPLES, normalize_parameters

DSS = np.array(next(iter(EXAMPLES.values()))["parameters"])
HOMOTAURINE = np.array(list(EXAMPLES.values())[1]["parameters"])
INPUT_NAMES = ("xA", "xX", "u", "v")


def physical_bounds(domain, reference=DSS):
    if domain == "local":
        return np.column_stack(
            (
                np.r_[reference[:3] - 0.3, 0.8 * reference[3:]],
                np.r_[reference[:3] + 0.3, 1.2 * reference[3:]],
            )
        )
    if domain == "broad":
        return np.array([[-15, -12.5]] * 3 + [[0.1, 7]] * 2)
    raise ValueError(domain)


def numerical_dataset(parameters):
    """Vectorized independent reference, preserving ascending physical order."""
    p = np.asarray(parameters, dtype=float)
    scale = -p[:, 1]
    reference = (-3 * p[:, 0] + p[:, 1] - 3 * p[:, 2]) / 4
    x = np.column_stack(
        (
            (p[:, 0] - p[:, 1]) / scale,
            (p[:, 2] - p[:, 1]) / scale,
            p[:, 3] / scale,
            p[:, 4] / scale,
        )
    )
    total = p[:, :3].sum(axis=1)
    H = np.zeros((len(p), 4, 4))
    for i in range(3):
        H[:, i, i] = -0.75 * total + p[:, i]
    H[:, 3, 3] = 0.25 * total
    for i, j, k in ((0, 1, 3), (1, 2, 4), (0, 3, 4), (2, 3, 3)):
        H[:, i, j] = H[:, j, i] = p[:, k] / 2
    E = np.linalg.eigvalsh(H)
    return dict(
        parameters_hz=p,
        features=x,
        targets=(E - reference[:, None]) / scale[:, None],
        scale_hz=scale,
        reference_hz=reference,
    )


def make_data(domain, n, seed=42, reference=DSS):
    rng = np.random.default_rng(seed)
    bounds = physical_bounds(domain, reference)
    data = numerical_dataset(rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 5)))
    order = rng.permutation(n)
    data.update(
        train_indices=order[: int(0.8 * n)],
        validation_indices=order[int(0.8 * n) : int(0.9 * n)],
        test_indices=order[int(0.9 * n) :],
    )
    return data


def gap_predictions(y, scales):
    """Fixed target indices: do not sort or repair learned outputs."""
    return (
        np.column_stack((y[:, 2] - y[:, 1], y[:, 3] - y[:, 2], y[:, 3] - y[:, 1]))
        * scales[:, None]
    )


def metrics(pred, target, scales, features):
    error = pred - target
    gaperror = gap_predictions(pred, scales) - gap_predictions(target, scales)
    denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    return dict(
        eigenvalue_rmse_dimensionless=float(np.sqrt(np.mean(error**2))),
        eigenvalue_rmse_hz=float(np.sqrt(np.mean((error * scales[:, None]) ** 2))),
        eigenvalue_max_abs_hz=float(np.max(np.abs(error * scales[:, None]))),
        eigenvalue_r2_per_branch=(
            1 - np.sum(error**2, axis=0) / np.maximum(denominator, 1e-30)
        ).tolist(),
        frequency_rmse_hz=float(np.sqrt(np.mean(gaperror**2))),
        frequency_max_abs_hz=float(np.max(abs(gaperror))),
        frequency_rmse_per_line_hz=np.sqrt(np.mean(gaperror**2, axis=0)).tolist(),
        ordering_violations=int(
            np.count_nonzero(np.any(np.diff(pred, axis=1) < 0, axis=1))
        ),
        samples=len(target),
        trace_max_abs_dimensionless=float(
            np.max(abs(pred.sum(axis=1) - (2 * (features[:, 0] + features[:, 1]) - 2)))
        ),
    )


def write_rows(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_predict(model, x, conditioning):
    xm, xs, ym, ys = conditioning
    with torch.no_grad():
        prediction = (
            model(torch.as_tensor((x - xm) / xs, dtype=next(model.parameters()).dtype))
            .cpu()
            .numpy()
        )
    return prediction * ys + ym


def train_one(data, seed, args, outdir, conditioning):
    torch.manual_seed(seed)
    xm, xs, ym, ys = conditioning
    train, val = data["train_indices"], data["validation_indices"]
    X = torch.tensor((data["features"][train] - xm) / xs, dtype=torch.float32)
    Y = torch.tensor((data["targets"][train] - ym) / ys, dtype=torch.float32)
    model = KharKAN((4, args.width, 4), include_abs=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_score = float("inf")
    best_state = None
    history = []
    start = time.perf_counter()
    switch = int(0.9 * args.epochs)
    for epoch in range(args.epochs):
        if epoch == switch:
            model.switch_to_sparsemax()
        order = torch.randperm(len(X))
        accumulated = 0
        for batch in order.split(args.batch_size):
            optimizer.zero_grad()
            pred = model(X[batch])
            fit = torch.mean(torch.abs(pred - Y[batch]))
            penalty = 0.1 * model.L05_loss() if epoch < switch else 0
            loss = fit + penalty
            loss.backward()
            optimizer.step()
            accumulated += float(fit.detach()) * len(batch)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            prediction = model_predict(model, data["features"][val], conditioning)
            score = float(np.sqrt(np.mean((prediction - data["targets"][val]) ** 2)))
            history.append(
                dict(
                    epoch=epoch + 1,
                    training_mae_standardized=accumulated / len(X),
                    validation_rmse_dimensionless=score,
                    sparsemax=epoch >= switch,
                )
            )
            if epoch >= switch and score < best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        if epoch % 100 == 0 or epoch == args.epochs - 1:
            print(
                f"{args.domain} seed={seed} epoch={epoch + 1}/{args.epochs} val_RMSE={score:.6g} elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    model.load_state_dict(best_state)
    weights = [layer._activation_weights().detach().numpy() for layer in model.layers]
    training_seconds = time.perf_counter() - start
    checkpoint = dict(
        state_dict=model.state_dict(),
        architecture=[4, args.width, 4],
        include_abs=False,
        sparsemax=True,
        input_mean=xm.tolist(),
        input_std=xs.tolist(),
        output_mean=ym.tolist(),
        output_std=ys.tolist(),
        seed=seed,
        best_epoch=best_epoch,
        validation_rmse_dimensionless=best_score,
        training_seconds=training_seconds,
    )
    torch.save(checkpoint, outdir / f"seed_{seed}.pt")
    write_rows(outdir / f"history_seed_{seed}.csv", history)
    summary = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    summary["activation_weights"] = [w.tolist() for w in weights]
    summary["support_sizes"] = [
        np.count_nonzero(w > 1e-8, axis=1).tolist() for w in weights
    ]
    summary["selected_activations"] = [
        [layer.activation_reprs[int(np.argmax(row))] for row in w]
        for layer, w in zip(model.layers, weights)
    ]
    return model, summary


def powers(degree):
    return [
        p for p in itertools.product(range(degree + 1), repeat=4) if sum(p) <= degree
    ]


def design_matrix(z, exponents):
    return np.array([np.prod(z**p, axis=1) for p in exponents]).T


def extract_formulas(model, conditioning, outdir, data, reference=DSS):
    """Compose the actual trained network and undo all affine conditioning."""
    xm, xs, ym, ys = conditioning
    model = model.double()
    raw = model.symbolic_formula(clean=False, round_digits=16)
    symbols = sp.symbols("xA xX u v")
    substitutions = {
        sp.Symbol(f"x_{j}"): (symbols[j] - float(xm[j])) / float(xs[j])
        for j in range(4)
    }
    expressions = [
        sp.Poly(
            sp.expand(
                raw[f"z_{i}"].subs(substitutions, simultaneous=True) * float(ys[i])
                + float(ym[i])
            ),
            *symbols,
        ).as_expr()
        for i in range(4)
    ]
    rounded = []
    for expr in expressions:
        rounded.append(
            sp.Add(
                *[
                    sp.Float(round(float(c), 10))
                    * sp.prod(x**k for x, k in zip(symbols, p))
                    for p, c in sp.Poly(expr, *symbols).terms()
                    if round(float(c), 10) != 0
                ]
            )
        )
    evaluator = sp.lambdify(symbols, expressions, "numpy", cse=True)
    x = data["features"][data["validation_indices"][:1000]]
    symbolic = np.asarray(evaluator(*x.T)).T
    numerical = model_predict(model, x, conditioning)
    discrepancy = float(np.max(abs(symbolic - numerical)))
    if discrepancy > 1e-8:
        raise AssertionError(f"Extraction disagrees with network: {discrepancy}")
    t = sp.Symbol("t")
    xstar, scale, _ = normalize_parameters(*reference)
    section_subs = dict(zip(symbols, [xstar[0], xstar[1], xstar[2] * t, xstar[3] * t]))
    section_lambda = [sp.expand(expr.subs(section_subs)) for expr in expressions]
    section_f = [
        sp.expand(scale * (section_lambda[j] - section_lambda[i]))
        for i, j in ((1, 2), (2, 3), (1, 3))
    ]
    tcoeffs = [[float(sp.Poly(f, t).nth(i)) for i in range(5)] for f in section_f]
    result = dict(
        input_names=list(INPUT_NAMES),
        formulas=[str(e) for e in expressions],
        rounded_formulas=[str(e) for e in rounded],
        latex=[sp.latex(e) for e in rounded],
        degree=[int(sp.Poly(e, *symbols).total_degree()) for e in expressions],
        terms=[len(sp.Poly(e, *symbols).terms()) for e in expressions],
        operations=[int(sp.count_ops(e)) for e in expressions],
        max_extraction_error_dimensionless=discrepancy,
        cross_section_coefficients_hz=tcoeffs,
        cross_section_latex=[sp.latex(e.evalf(10)) for e in section_f],
    )
    (outdir / "learned_formulas.json").write_text(json.dumps(result, indent=2) + "\n")
    with (outdir / "learned_formulas.tex").open("w") as stream:
        stream.write(
            "% Extracted from trained nmrkan.models.KharKAN; not an analytical fit.\n"
        )
        for i, expr in enumerate(rounded):
            stream.write(
                "\\[\\widehat\\lambda_{" + str(i) + "}=" + sp.latex(expr) + "\\]\n"
            )
        for label, expr in zip((12, 23, 13), section_f):
            stream.write(
                "\\[\\widehat f_{"
                + str(label)
                + "}(t)="
                + sp.latex(expr.evalf(10))
                + "\\]\n"
            )
    return result, expressions


def run(args):
    if args.samples < 20 or args.epochs < 2 or args.batch_size < 1 or args.width < 1:
        raise ValueError(
            "Need at least 20 samples, 2 epochs, and positive batch size/width."
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    molecule = getattr(args, "molecule", "dss")
    reference = DSS if molecule == "dss" else HOMOTAURINE
    outdir = ROOT / molecule / args.domain
    if (outdir / "config.json").exists():
        raise FileExistsError(
            f"{outdir} already contains a run; choose a fresh NMRKAN_MOLECULE_OUTPUT."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    data = make_data(args.domain, args.samples, reference=reference)
    np.savez_compressed(outdir / "dataset.npz", **data)
    train, val, test = (
        data["train_indices"],
        data["validation_indices"],
        data["test_indices"],
    )
    X, Y = data["features"], data["targets"]
    conditioning = (
        X[train].mean(axis=0),
        X[train].std(axis=0),
        Y[train].mean(axis=0),
        Y[train].std(axis=0),
    )
    config = dict(
        molecule=molecule,
        reference_parameters_hz=reference.tolist(),
        domain=args.domain,
        bounds_hz=physical_bounds(args.domain, reference).tolist(),
        samples=args.samples,
        split_counts=[len(train), len(val), len(test)],
        data_seed=42,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        architecture=[4, args.width, 4],
        activation_library=["x", "x^2", "0"],
        learning_rate=0.001,
        optimizer="Adam",
        fit_loss="MAE on train-standardized normalized eigenvalues",
        regularization=0.1,
        softmax_fraction=0.9,
        selection="minimum validation RMSE in unstandardized dimensionless eigenvalues, sparsemax phase only",
        input_names=list(INPUT_NAMES),
        conditioning=[z.tolist() for z in conditioning],
        target_source="numpy.linalg.eigvalsh of physical 4x4 Hamiltonian, ascending, then shift/scale",
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        device="CPU",
        threads=1,
        model_sha256=hashlib.sha256(
            (PROJECT_ROOT / "nmrkan/models.py").read_bytes()
        ).hexdigest(),
        dataset_sha256=hashlib.sha256(
            (outdir / "dataset.npz").read_bytes()
        ).hexdigest(),
    )
    (outdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    models = []
    summaries = []
    for seed in args.seeds:
        model, summary = train_one(data, seed, args, outdir, conditioning)
        models.append(model)
        summaries.append(summary)
        (outdir / "training_summaries.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
    winner = min(
        range(len(models)), key=lambda i: summaries[i]["validation_rmse_dimensionless"]
    )
    selected = models[winner]
    # The test set is first evaluated only after all training and model selection.
    for model, summary in zip(models, summaries):
        pred = model_predict(model, X[test], conditioning)
        summary["test"] = metrics(pred, Y[test], data["scale_hz"][test], X[test])
    (outdir / "training_summaries.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    formulas, expressions = extract_formulas(
        selected, conditioning, outdir, data, reference
    )
    prediction = model_predict(selected, X[test], conditioning)
    evaluations = {"ASN": metrics(prediction, Y[test], data["scale_hz"][test], X[test])}
    np.savez_compressed(
        outdir / "heldout_predictions.npz",
        test_indices=test,
        exact=Y[test],
        asn=prediction,
    )
    xm, xs, _, _ = conditioning
    baselines = {}
    for degree in (2, 4):
        exp = powers(degree)
        start = time.perf_counter()
        coeff = np.linalg.lstsq(
            design_matrix((X[train] - xm) / xs, exp), Y[train], rcond=None
        )[0]
        seconds = time.perf_counter() - start
        pred = design_matrix((X[test] - xm) / xs, exp) @ coeff
        baselines[degree] = (exp, coeff)
        evaluations[f"OLS degree {degree}"] = metrics(
            pred, Y[test], data["scale_hz"][test], X[test]
        )
        evaluations[f"OLS degree {degree}"].update(
            training_seconds=seconds, basis_terms=len(exp)
        )
        np.savez(
            outdir / f"ols_degree_{degree}.npz",
            exponents=exp,
            coefficients=coeff,
            input_mean=xm,
            input_std=xs,
        )
    rounded_eval = sp.lambdify(
        sp.symbols("xA xX u v"),
        [sp.sympify(e) for e in formulas["rounded_formulas"]],
        "numpy",
        cse=True,
    )
    rounded = np.asarray(rounded_eval(*X[test].T)).T
    evaluations["ASN coefficients rounded to 10 decimals"] = metrics(
        rounded, Y[test], data["scale_hz"][test], X[test]
    )
    t = np.linspace(0, 1.5, 601)
    p = np.tile(reference, (len(t), 1))
    p[:, 3:] *= t[:, None]
    section = numerical_dataset(p)
    section_pred = model_predict(selected, section["features"], conditioning)
    exactf = gap_predictions(section["targets"], section["scale_hz"])
    asnf = gap_predictions(section_pred, section["scale_hz"])
    coefficients = np.array(formulas["cross_section_coefficients_hz"])
    np.testing.assert_allclose(
        np.polynomial.polynomial.polyval(t, coefficients.T).T, asnf, atol=1e-8
    )
    olsf = {
        degree: gap_predictions(
            design_matrix((section["features"] - xm) / xs, exp) @ coeff,
            section["scale_hz"],
        )
        for degree, (exp, coeff) in baselines.items()
    }
    rows = []
    for k, tk in enumerate(t):
        row = dict(t=tk)
        for i in range(4):
            row[f"exact_lambda{i}"] = section["targets"][k, i]
            row[f"asn_lambda{i}"] = section_pred[k, i]
        for j, label in enumerate((12, 23, 13)):
            row[f"exact_f{label}_hz"] = exactf[k, j]
            row[f"asn_f{label}_hz"] = asnf[k, j]
            for degree in (2, 4):
                row[f"ols{degree}_f{label}_hz"] = olsf[degree][k, j]
        rows.append(row)
    write_rows(outdir / "cross_section_comparison.csv", rows)
    idx = np.argmin(abs(t - 1))
    bounds = physical_bounds(args.domain, reference)
    inside = np.all((p >= bounds[:, 0] - 1e-12) & (p <= bounds[:, 1] + 1e-12), axis=1)
    # Boundary challenge: outside the local domain; held-out boundaries of broad domain.
    rng = np.random.default_rng(12345)
    boundary = rng.uniform(
        physical_bounds("broad")[:, 0], physical_bounds("broad")[:, 1], size=(2000, 5)
    )
    boundary[:1000, 3] = 0
    boundary[1000:, 4] = 0
    boundary_data = numerical_dataset(boundary)
    boundary_pred = model_predict(selected, boundary_data["features"], conditioning)
    result = dict(
        config=config,
        selected_seed=summaries[winner]["seed"],
        selected_epoch=summaries[winner]["best_epoch"],
        evaluations=evaluations,
        seeds=summaries,
        formulas=formulas,
        reference_point=dict(
            exact_frequencies_hz=exactf[idx].tolist(),
            asn_frequencies_hz=asnf[idx].tolist(),
            error_hz=(asnf[idx] - exactf[idx]).tolist(),
            ols2_frequencies_hz=olsf[2][idx].tolist(),
            ols4_frequencies_hz=olsf[4][idx].tolist(),
        ),
        section_in_domain=dict(
            t_min=float(t[inside].min()),
            t_max=float(t[inside].max()),
            frequency_rmse_hz=float(
                np.sqrt(np.mean((asnf[inside] - exactf[inside]) ** 2))
            ),
            frequency_max_abs_hz=float(np.max(abs(asnf[inside] - exactf[inside]))),
        ),
        section_outside_domain=dict(
            frequency_rmse_hz=float(
                np.sqrt(np.mean((asnf[~inside] - exactf[~inside]) ** 2))
            ),
            frequency_max_abs_hz=float(np.max(abs(asnf[~inside] - exactf[~inside]))),
        ),
        boundary_challenge=metrics(
            boundary_pred,
            boundary_data["targets"],
            boundary_data["scale_hz"],
            boundary_data["features"],
        ),
    )
    if molecule == "dss":
        result["dss_point"] = result["reference_point"]
    (outdir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "DONE",
        molecule,
        args.domain,
        "selected seed",
        result["selected_seed"],
        "test",
        evaluations["ASN"],
        "reference",
        result["reference_point"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=("dss", "homotaurine"), default="dss")
    parser.add_argument("--domain", choices=("local", "broad"), required=True)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    run(parser.parse_args())
