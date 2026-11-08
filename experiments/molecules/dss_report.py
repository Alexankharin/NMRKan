"""Evaluate trained ASN artifacts, plot comparisons, and write paper-ready reports.

Run only after both train_dss_asn.py domains finish. This script does not train,
refit, or select a network using test data. It evaluates the saved selection.
"""

import base64
import csv
import json
import math
import re

import markdown
import numpy as np
import sympy as sp
from latex2mathml.converter import convert

from nmrkan.physics.three_ch2 import hamiltonian, normalize_parameters
from nmrkan.physics.real_molecules import (
    analytical_eigenvalues,
    cross_section_taylor,
    effective_three_state,
    singlet_observable,
    spectral_lines,
    lorentzian_spectrum,
)
from experiments.molecules.train import (
    ROOT,
    DSS,
    gap_predictions,
    metrics,
)

FOLDER = ROOT / "dss"
SYMBOLS = sp.symbols("xA xX u v")


def formula_values(expressions, x):
    fn = sp.lambdify(SYMBOLS, expressions, "numpy", cse=True)
    values = fn(*x.T)
    return np.column_stack([np.broadcast_to(v, (len(x),)) for v in values])


def analyze(domain, folder_root=FOLDER, reference=DSS):
    folder = folder_root / domain
    result = json.loads((folder / "comparison.json").read_text())
    ds = np.load(folder / "dataset.npz")
    i = ds["test_indices"]
    p = ds["parameters_hz"][i]
    x = ds["features"][i]
    y = ds["targets"][i]
    s = ds["scale_hz"][i]
    r = ds["reference_hz"][i]
    exact = np.array([analytical_eigenvalues(row) for row in p])
    evaluations = result["evaluations"].copy()
    evaluations["Exact analytical cubic"] = metrics(
        (exact - r[:, None]) / s[:, None], y, s, x
    )
    for name, method in [
        ("Bare 3x3", lambda p: hamiltonian(*p)[:3, :3]),
        ("Effective 3x3", effective_three_state),
    ]:
        principal = np.array([np.linalg.eigvalsh(method(row)) for row in p])
        # Only principal branches exist for these approximations: report gaps only.
        gap = np.column_stack(
            (
                principal[:, 1] - principal[:, 0],
                principal[:, 2] - principal[:, 1],
                principal[:, 2] - principal[:, 0],
            )
        )
        err = gap - gap_predictions(y, s)
        evaluations[name] = dict(
            frequency_rmse_hz=float(np.sqrt(np.mean(err**2))),
            frequency_max_abs_hz=float(np.max(abs(err))),
        )
    original = [sp.sympify(e) for e in result["formulas"]["formulas"]]
    paper = []
    for expression in original:
        paper.append(
            sp.Add(
                *[
                    sp.Float(round(float(c), 3))
                    * sp.prod(v**k for v, k in zip(SYMBOLS, power))
                    for power, c in sp.Poly(expression, *SYMBOLS).terms()
                    if abs(float(c)) >= 1e-3 and round(float(c), 3) != 0
                ]
            )
        )
    rounded = formula_values(paper, x)
    evaluations["ASN paper-style 3-decimal rounding"] = metrics(rounded, y, s, x)
    xp, scale, _ = normalize_parameters(*reference)
    dss_rounded = gap_predictions(
        formula_values(paper, xp[None, :]), np.array([scale])
    )[0]
    result["evaluations_extended"] = evaluations
    result["paper_formulas"] = [str(e) for e in paper]
    result["paper_dss_frequencies_hz"] = dss_rounded.tolist()
    result["paper_reference_frequencies_hz"] = dss_rounded.tolist()
    if result["config"].get("molecule") == "homotaurine":
        del result["paper_dss_frequencies_hz"]
    result["seed_statistics"] = {}
    for key in (
        "eigenvalue_rmse_dimensionless",
        "frequency_rmse_hz",
        "frequency_max_abs_hz",
    ):
        a = np.array([seed["test"][key] for seed in result["seeds"]])
        result["seed_statistics"][key] = dict(
            mean=float(a.mean()),
            std=float(a.std(ddof=1)),
            min=float(a.min()),
            max=float(a.max()),
        )
    (folder / "postprocessing.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "paper_rounded_formulas.tex").open("w") as stream:
        stream.write(
            "% Actual ASN extraction; discard |coefficient|<0.001, round remaining to 3 decimals.\n"
        )
        for j, e in enumerate(paper):
            stream.write(
                "\\[\\widehat\\lambda_{" + str(j) + "}=" + sp.latex(e) + "\\]\n"
            )
    return result


def section(domain, folder_root=FOLDER):
    rows = list(
        csv.DictReader((folder_root / domain / "cross_section_comparison.csv").open())
    )
    return {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}


def figures(results, folder_root=FOLDER, reference=DSS, molecule="DSS"):
    FOLDER = folder_root
    DSS = reference
    slug = molecule.lower()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = ["#0077aa", "#d1495b", "#308050", "#7851a9"]

    def save(fig, name):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FOLDER / f"{name}.{ext}", dpi=170)
        plt.close(fig)

    local = results["local"]
    data = section("local", FOLDER)
    t = data["t"]
    inside = (t >= 0.8 - 1e-10) & (t <= 1.2 + 1e-10)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for k, color in enumerate(colors):
        axes[0, 0].plot(
            t[inside],
            data[f"exact_lambda{k}"][inside],
            color=color,
            lw=2,
            label=rf"$\lambda_{k}$ exact",
        )
        axes[0, 0].plot(
            t[inside], data[f"asn_lambda{k}"][inside], color=color, ls="--", lw=1.5
        )
    axes[0, 0].set(
        xlabel="Section parameter t",
        ylabel="Dimensionless eigenvalue",
        title="A  Ordered eigenvalues: exact and learned",
    )
    axes[0, 0].legend(ncols=2, fontsize=8)
    for k, label in enumerate((12, 23, 13)):
        axes[0, 1].plot(
            t[inside],
            data[f"exact_f{label}_hz"][inside],
            color=colors[k],
            label=rf"$f_{{{label}}}$ exact",
        )
        axes[0, 1].plot(
            t[inside], data[f"asn_f{label}_hz"][inside], color=colors[k], ls="--"
        )
        axes[1, 0].plot(
            t[inside],
            1000 * (data[f"asn_f{label}_hz"] - data[f"exact_f{label}_hz"])[inside],
            color=colors[k],
            label=rf"$f_{{{label}}}$ ASN",
        )
        axes[1, 0].plot(
            t[inside],
            1000 * (data[f"ols2_f{label}_hz"] - data[f"exact_f{label}_hz"])[inside],
            color=colors[k],
            ls=":",
        )
    axes[0, 1].set(
        xlabel="Section parameter t",
        ylabel="ZQ frequency / Hz",
        title="B  Solid: analytical; dashed: trained ASN",
    )
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].axhline(0, color="0.7", lw=0.8)
    axes[1, 0].set(
        xlabel="Section parameter t",
        ylabel="Predicted minus exact / mHz",
        title="C  Solid: ASN; dotted: quadratic least squares",
    )
    axes[1, 0].legend(fontsize=8)
    O = singlet_observable("A")
    _, all_lines = spectral_lines(hamiltonian(*DSS), O, O)
    lines = [line for line in all_lines if line["i"] > 0]
    order = {(1, 2): 0, (2, 3): 1, (1, 3): 2}
    point = local.get("reference_point", local.get("dss_point"))
    learned = [
        dict(
            line,
            frequency_hz=point["asn_frequencies_hz"][order[(line["i"], line["j"])]],
        )
        for line in lines
    ]
    centers = point["exact_frequencies_hz"]
    freq = np.linspace(max(0, min(centers) - 0.6), max(centers) + 0.6, 12001)
    yy = lorentzian_spectrum(freq, lines)
    yp = lorentzian_spectrum(freq, learned)
    axes[1, 1].plot(freq, yy / yy.max(), color="#1d3557", label="Exact centers")
    axes[1, 1].plot(freq, yp / yy.max(), color="#e76f51", ls="--", label="ASN centers")
    axes[1, 1].set(
        xlabel="ZQ frequency / Hz",
        ylabel="Common amplitude scale",
        title=f"D  {molecule} spectrum; shared illustrative amplitudes",
    )
    axes[1, 1].legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axvline(1, color="0.65", ls=":")
    fig.suptitle(
        f"{molecule}: actual trained ASN versus exact analytical reference\nFour-input model; selected seed {local['selected_seed']}; solid/dashed overlap at this scale",
        fontsize=13,
    )
    save(fig, f"{slug}_asn_vs_analytical")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    names = ["ASN", "OLS degree 2", "OLS degree 4", "Effective 3x3", "Bare 3x3"]
    positions = np.arange(len(names))
    for j, domain in enumerate(("local", "broad")):
        vals = [
            results[domain]["evaluations_extended"][k]["frequency_rmse_hz"]
            for k in names
        ]
        axes[0].bar(
            positions + (j - 0.5) * 0.35,
            vals,
            width=0.35,
            label=domain,
            color=colors[j],
        )
    axes[0].set(
        yscale="log",
        ylabel="Held-out frequency RMSE / Hz",
        title="A  Same held-out samples within each domain",
    )
    axes[0].set_xticks(
        positions, ["ASN", "OLS 2", "OLS 4", "Effective\n3×3", "Bare\n3×3"]
    )
    axes[0].legend()
    for j, domain in enumerate(("local", "broad")):
        d = section(domain, FOLDER)
        err = np.max(
            np.abs(
                np.column_stack(
                    [d[f"asn_f{k}_hz"] - d[f"exact_f{k}_hz"] for k in (12, 23, 13)]
                )
            ),
            axis=1,
        )
        axes[1].plot(d["t"], err, color=colors[j], label=f"{domain} ASN")
    axes[1].axvspan(
        0.8, 1.2, color=colors[0], alpha=0.08, label="Local section training range"
    )
    broad_max = 7 / max(DSS[3:])
    if broad_max <= 1.5:
        axes[1].axvline(
            broad_max,
            color=colors[1],
            ls="-.",
            lw=0.8,
            label="Broad section upper limit",
        )
    axes[1].axhline(0.1, color="0.4", ls=":", label="Illustrative 0.1 Hz linewidth")
    axes[1].set(
        yscale="log",
        xlabel="Section parameter t",
        ylabel="Largest line error / Hz",
        title="B  Errors along and beyond training sections",
    )
    axes[1].legend(fontsize=8)
    save(fig, f"{slug}_asn_domain_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, domain in zip(axes, ("local", "broad")):
        for seed in results[domain]["seeds"]:
            rows = list(
                csv.DictReader(
                    (FOLDER / domain / f"history_seed_{seed['seed']}.csv").open()
                )
            )
            ax.plot(
                [int(r["epoch"]) for r in rows],
                [float(r["validation_rmse_dimensionless"]) for r in rows],
                label=str(seed["seed"]),
            )
        ax.axvline(900, color="black", ls=":", label="Sparsemax starts")
        ax.set(
            yscale="log",
            xlabel="Epoch",
            ylabel="Validation RMSE (dimensionless)",
            title=domain.capitalize() + " domain: five independent initializations",
        )
        ax.legend(fontsize=8, ncols=2)
    save(fig, f"{slug}_asn_training")


def metric_table(result):
    text = "| Method | Eigenvalue RMSE / Hz | Frequency RMSE / Hz | Max. line error / Hz | Ordering violations |\n|---|---:|---:|---:|---:|\n"
    for name in (
        "Exact analytical cubic",
        "ASN",
        "ASN paper-style 3-decimal rounding",
        "OLS degree 2",
        "OLS degree 4",
        "Effective 3x3",
        "Bare 3x3",
    ):
        row = result["evaluations_extended"][name]
        energy = (
            f"{row['eigenvalue_rmse_hz']:.6g}" if "eigenvalue_rmse_hz" in row else "—"
        )
        text += f"| {name} | {energy} | {row['frequency_rmse_hz']:.6g} | {row['frequency_max_abs_hz']:.6g} | {row.get('ordering_violations', '—')} |\n"
    return text


def build_text(results):
    local, broad = results["local"], results["broad"]
    lf = local["formulas"]
    dss = local["dss_point"]
    exactt = cross_section_taylor(DSS)
    coeff = np.array(lf["cross_section_coefficients_hz"])
    t = sp.Symbol("t")
    polynomial_lines = []
    for label, c in zip((12, 23, 13), coeff):
        expr = sum(
            sp.Float(round(float(value), 9)) * t**k
            for k, value in enumerate(c)
            if abs(value) > 1e-10
        )
        polynomial_lines.append(
            "$$\\widehat f_{"
            + str(label)
            + "}(t)="
            + sp.latex(expr)
            + "\\quad\\mathrm{Hz}.$$"
        )
    center_table = "| Line | Exact / Hz | ASN / Hz | ASN error / mHz | Quadratic OLS / Hz |\n|---|---:|---:|---:|---:|\n"
    for k, label in enumerate((12, 23, 13)):
        center_table += f"| f{label} | {dss['exact_frequencies_hz'][k]:.9f} | {dss['asn_frequencies_hz'][k]:.9f} | {1000 * dss['error_hz'][k]:+.6f} | {dss['ols2_frequencies_hz'][k]:.9f} |\n"
    derivative_table = "| Line | Exact value | Learned value | Exact slope | Learned slope | Exact quadratic coefficient | Learned quadratic coefficient |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for k, label in enumerate((12, 23, 13)):
        learned = np.array(
            [
                sum(c * math.comb(n, j) for n, c in enumerate(coeff[k]) if n >= j)
                for j in range(3)
            ]
        )
        derivative_table += f"| f{label} | {exactt[k, 0]:.6f} | {learned[0]:.6f} | {exactt[k, 1]:.6f} | {learned[1]:.6f} | {exactt[k, 2]:.6f} | {learned[2]:.6f} |\n"
    seed_table = "| Domain | Seed | Best epoch | Validation eigenvalue RMSE (dimensionless) | Test frequency RMSE / Hz |\n|---|---:|---:|---:|---:|\n"
    for domain, result in results.items():
        for seed in result["seeds"]:
            mark = " (selected)" if seed["seed"] == result["selected_seed"] else ""
            seed_table += f"| {domain} | {seed['seed']}{mark} | {seed['best_epoch']} | {seed['validation_rmse_dimensionless']:.7g} | {seed['test']['frequency_rmse_hz']:.7g} |\n"
    selection = next(s for s in local["seeds"] if s["seed"] == local["selected_seed"])
    choices = selection["selected_activations"]
    labels = {"": "identity", "**2": "square", "*0": "zero"}
    support = "; ".join(
        f"layer {i + 1}: " + ", ".join(f"{labels[k]}={part.count(k)}" for k in labels)
        for i, part in enumerate(choices)
    )
    monomials = sorted(
        set(
            power
            for expression in local["paper_formulas"]
            for power, _ in sp.Poly(sp.sympify(expression), *SYMBOLS).terms()
        ),
        key=lambda p: (sum(p), p),
        reverse=False,
    )
    coef_table = "| Monomial | λ0 | λ1 | λ2 | λ3 |\n|---|---:|---:|---:|---:|\n"
    polys = [sp.Poly(sp.sympify(e), *SYMBOLS) for e in local["paper_formulas"]]
    for power in monomials:
        term = sp.prod(s**k for s, k in zip(SYMBOLS, power))
        coef_table += (
            "| $"
            + sp.latex(term, symbol_names={SYMBOLS[0]: r"x_A", SYMBOLS[1]: r"x_X"})
            + "$ | "
            + " | ".join(f"{float(poly.coeff_monomial(term)):.3f}" for poly in polys)
            + " |\n"
        )
    localstats = local["seed_statistics"]["frequency_rmse_hz"]
    broadstats = broad["seed_statistics"]["frequency_rmse_hz"]
    source = r"""
# DSS: formulas learned by ASN, compared with the analytical solution

Actual training of the repository KAN/ASN · 18 October 2026

**The learned result is now separate from the analytical reference.** We trained the existing `nmrkan.models.KharKAN` implementation on numerically diagonalized DSS-type Hamiltonians, extracted its algebraic formulas, and evaluated them against the independent exact solution. The network never received the cubic formula, normal-mode energies, analytic derivatives, or a fitted one-dimensional cross-section as training inputs. It learned four ordered normalized eigenvalues from four normalized Hamiltonian parameters.

The manuscript calls this architecture an **activation-selection network (ASN)**. The repository class name is `KharKAN`; this experiment uses that class and its learned mixtures of zero, identity, and square functions. It is not a spline-edge KAN. The earlier [analytical report](dss_analysis.html) supplies the physical derivation and approximation audit; this report supplies the missing learned result and comparison.

## 1. Main DSS result

@@POINT@@

The selected local ASN has held-out frequency RMSE **@@LOCAL_RMSE@@ Hz**, maximum held-out line error **@@LOCAL_MAX@@ Hz**, and a maximum line error at the supplied DSS point of **@@POINT_MAX@@ Hz**. These errors measure approximation to the specified nearest-neighbor secular Hamiltonian; they are not experimental residuals. The predicted spectral positions are therefore usable within the demonstrated domain, subject to the separate physical approximations documented in the analytical report.

![Learned ASN and exact analytical cross-section](dss/dss_asn_vs_analytical.png)

## 2. Training design and correspondence to the manuscript

We used the full four-state matrix, including type-II mixing, with $q=0$. Targets were generated by `numpy.linalg.eigvalsh` of the physical Hamiltonian, ordered from lowest to highest, then transformed as $\lambda_i=(E_i-E_{\rm ref})/s$. The four inputs are

$$s=-J_M,\quad x_A=(J_A-J_M)/s,\quad x_X=(J_X-J_M)/s,\quad u=d/s,\quad v=e/s.$$

All four energies are learned jointly. No positive-energy filtering, post-prediction sorting, symmetry averaging, or trace projection is applied. Frequency predictions use fixed target indices:

$$\widehat E_i=E_{\rm ref}+s\widehat\lambda_i,\qquad
\widehat f_{12}=s(\widehat\lambda_2-\widehat\lambda_1),\quad
\widehat f_{23}=s(\widehat\lambda_3-\widehat\lambda_2),\quad
\widehat f_{13}=\widehat f_{12}+\widehat f_{23}.$$

Two parameter domains were specified before evaluating the test data. The **DSS local domain** independently varies each supplied geminal coupling by ±0.3 Hz and each nearest-neighbor difference by ±20%. It is a numerical approximation domain, not a fitted uncertainty interval. The **broad domain** is the original supplied box: three geminals in [−15,−12.5] Hz and both differences in [0.1,7] Hz. The local e range extends to 7.9166604 Hz, so these two domains are deliberately different; their errors are not compared as if measured on identical populations.

Within each domain, 100,000 physical parameter vectors were sampled using NumPy seed 42, then divided into **80,000 training, 10,000 validation, and 10,000 test samples** with disjoint indices. Model initialization and minibatch-order seeds were 42, 43, 44, 45, and 46. The exact supplied DSS point and the displayed continuous section were not inserted into the training set.

The architecture is **(4,16,4)** with biases and activation library **{0,x,x²}**. This directly extends the manuscript's (2,16,4) network to four inputs. Training used Adam, learning rate 0.001, batch size 4096, 1000 epochs, and MAE. For epochs 1–900, softmax activation weights and a 0.1×L0.5 regularizer were used; epochs 901–1000 used sparsemax without that regularizer. Each seed's checkpoint and the representative seed were selected only by validation RMSE in dimensionless eigenvalues during the sparsemax phase. Test data were evaluated after this selection.

One additional numerical-conditioning step differs from the original manuscript experiment: inputs and normalized outputs were centered and divided by their **training-set** standard deviations. MAE was applied to these standardized outputs. Both affine transforms were algebraically undone during extraction, so the delivered formulas are functions of $(x_A,x_X,u,v)$ and return $\lambda_i$ in the stated physical normalization. This is a new DSS experiment, not a claim to reproduce the paper's older training run bit for bit.

The selected local model uses seed **@@LOCAL_SEED@@**, checkpoint epoch **@@LOCAL_EPOCH@@**. Its activation choices are **@@ACTIVATIONS@@**. The selected model's extracted polynomial degrees are **@@DEGREES@@**. Sparsemax support sizes, weights, optimizer settings, data hashes, model-source hash, versions, full training curves, and every checkpoint are retained in `dss/`.

## 3. The actual learned formulas

The files `dss/local/learned_formulas.json` and `.tex` contain the full four-input formulas obtained by composing the trained layers. They are not a least-squares fit substituted for the network. Evaluation of the extracted formulas agrees with the trained network to **@@EXTRACTION@@ dimensionless units** on 1000 validation points. An independent test reloads the saved checkpoint and checks this equivalence on held-out inputs.

For the paper-style representation, coefficients with magnitude below 0.001 were removed and remaining coefficients rounded to three decimals, following the supplied manuscript's extraction procedure. Unequal geminals do not permit the manuscript's uniform-chain symmetry averaging, so that operation was not applied. The table below defines $\widehat\lambda_i=\sum_m c_{im}m(x_A,x_X,u,v)$ for the **rounded local model**. Full-precision and rounded errors are reported separately.

@@COEFFICIENTS@@

These are learned coefficients. The exact analytical cubic has a different role: it is a physics reference used to assess the approximation. Ordinary least squares below is another distinct model trained on the same data; it is never labeled ASN.

## 4. Analytical and learned expressions on the same cross-section

Fix the DSS geminals and use

$$d(t)=5.269170t,\qquad e(t)=6.597217t\quad\mathrm{Hz}.$$

The exact analytical construction uses the squared normal-mode roots of

$$z^3-C_1(t)z^2+C_2(t)z-C_3(t)=0,$$

$$C_1(t)=3.162281774033+0.396528696796t^2,$$
$$C_2(t)=3.330354094093+0.426234440375t^2+0.037387752092t^4,\quad C_3(t)=1.168072320061.$$

For ordered roots $z_1\le z_2\le z_3$, define $\omega_i=s\sqrt{z_i}$. Then

$$f_{12}=\omega_3-\omega_2,\quad f_{23}=\omega_2-\omega_1,\quad f_{13}=\omega_3-\omega_1.$$

The **learned** expressions below are obtained solely by substituting this section into the trained four-input ASN formulas and taking their differences. No additional fitting along t was performed. Displayed coefficients are rounded to nine decimals; the stored coefficients retain numerical precision.

@@POLYNOMIALS@@

These learned expressions are finite-domain approximations. Their range of support from the local training domain is $0.8\le t\le1.2$. On the 161-point sampled section within that interval, the learned frequency RMSE is **@@SECTION_RMSE@@ Hz** and maximum error is **@@SECTION_MAX@@ Hz**. Results outside this interval are explicitly treated as extrapolation.

For a direct comparison with the analytical local description, expand about $\delta=t-1$. The columns below give the constant, linear, and quadratic coefficients in Hz. Analytical coefficients come from differentiating the exact Hamiltonian solution. Learned coefficients come from differentiating the extracted ASN polynomial; they were not imposed during training.

@@DERIVATIVES@@

Agreement in values need not imply equally accurate derivatives or curvatures. Unlike the exact Taylor series, the learned polynomial is optimized over a finite four-dimensional sample distribution. It must not be described as rediscovering perturbative coefficients merely because its polynomial degree matches.

## 5. Held-out comparisons and seed variability

All methods within a domain use the same 80,000 training vectors and 10,000 held-out test vectors. The fixed-basis fits use complete degree-2 (15 monomials) and degree-4 (70 monomials) polynomial bases in the same affinely conditioned inputs, and ordinary least squares. Degree 2 is the natural matched polynomial baseline when the learned ASN reduces to a quadratic; degree 4 tests additional polynomial capacity. Least squares minimizes squared error, whereas ASN training uses MAE plus activation regularization. The physics references do not fit training data.

**DSS local domain**

@@LOCAL_TABLE@@

**Broad supplied domain**

@@BROAD_TABLE@@

For the selected broad model, the DSS-point predicted frequencies are **@@BROAD_POINT@@ Hz**. Its broader-domain performance must not be conflated with the local model's accuracy. Separate checkpoint files and reports identify which domain produced each prediction.

Five-seed frequency-RMSE means ± sample standard deviations are **@@LOCAL_MEAN@@ ± @@LOCAL_SD@@ Hz** locally and **@@BROAD_MEAN@@ ± @@BROAD_SD@@ Hz** broadly. These statistics measure optimization variability on fixed data, not experimental uncertainty or generalization over independently regenerated datasets.

@@SEEDS@@

![Held-out model and domain comparisons](dss/dss_asn_domain_comparison.png)

![Training curves for all seeds](dss/dss_asn_training.png)

## 6. Ordering, rounding, and observability

The training labels use the manuscript's ordered physical-energy convention. The exact solution's degeneracies and boundary cusps remain a challenge for a global polynomial approximation. We count negative adjacent predicted energy differences before any correction. The selected local and broad models have **@@LOCAL_VIOLATIONS@@** and **@@BROAD_VIOLATIONS@@** ordering violations, respectively, on their 10,000-point in-domain test sets.

A separate 2000-point decoupled-boundary challenge sets d=0 or e=0 with broad-box geminals. It is outside the local model's domain and on the closure of the broad model's domain. Local and broad maximum line errors there are **@@LOCAL_BOUNDARY@@ Hz** and **@@BROAD_BOUNDARY@@ Hz**, with **@@LOCAL_BOUNDARY_VIOLATIONS@@** and **@@BROAD_BOUNDARY_VIOLATIONS@@** ordering violations. These tests expose limitations; the analysis does not sort the predictions afterward to hide swaps. Neither test accuracy nor zero sampled violations is a mathematical global ordering guarantee.

The gap identity $\widehat f_{13}=\widehat f_{12}+\widehat f_{23}$ holds by construction, but no trace constraint was imposed on the four learned energies. Trace errors are recorded in the JSON metrics. The paper-style rounded local formulas predict DSS frequencies **@@ROUNDED_POINT@@ Hz**; their distinct error row quantifies information lost during symbolic presentation.

For the spectral overlay, both predictions use the same illustrative local-singlet autocorrelation amplitudes and a Lorentzian FWHM of 0.1 Hz. **Only the line centers are replaced by the ASN predictions.** This isolates the consequence of learned eigenvalue errors. The ASN does not predict eigenvectors, preparation efficiencies, relaxation, or experimental line intensities. The underlying molecular approximation retains type-II mixing and three unequal geminals; omitted fitted q still causes approximately 0.002 Hz line shifts at DSS, independently of surrogate error.

## 7. How this fits the paper's narrative

The physical example now has the required sequence: a real molecule supplies couplings; numerical diagonalization supplies supervised targets; dimensionless normalization reduces five physical parameters to four inputs; the ASN learns a sparse polynomial representation; algebraic extraction produces explicit four-input and cross-section formulas; and the predicted energy differences are compared with an exact analytical reference in spectral units.

The evidence supports **useful explicit learned approximations in a stated molecular domain**. It does not establish an accuracy advantage over a suitable polynomial basis or the exact cubic solver. The least-squares results must remain in the paper, including any cases where they outperform ASN. This is consistent with the manuscript's existing conclusion that activation selection does not intrinsically beat an appropriate predefined basis.

The broad-domain experiment and boundary checks make the limitation visible: a compact polynomial cannot generally reproduce every cusp of ordered eigenvalue surfaces. The local DSS demonstration should therefore be presented as a controlled molecular application, not a universal analytical solution. The exact result is the reference; the ASN result is the learned approximation. Training over a wider domain, using richer primitives, or adding constraints would be new experiments rather than corrections silently applied to these results.

The editor's separate request for comparisons with contemporary symbolic-regression methods still calls for a justified external baseline such as PySR/EQL and a common budget protocol. The current work runs ASN and fixed-basis baselines, not those external methods. Literature context and the experimental provenance limits remain documented in the analytical report. No experimental residuals or measured intensities have been invented.

## 8. Reproducibility and manuscript material

- `train_dss_asn.py` performs the actual training, saves all five seeds in each domain, and extracts the selected network.
- `dss/{local,broad}/dataset.npz` stores physical parameters, normalized inputs, ordered targets, and disjoint split indices.
- `config.json`, `training_summaries.json`, `history_seed_*.csv`, and `seed_*.pt` retain the complete experiment configuration and model artifacts.
- `learned_formulas.json`, `learned_formulas.tex`, and `paper_rounded_formulas.tex` distinguish full-precision extraction from paper-style rounding.
- `comparison.json`, `postprocessing.json`, `heldout_predictions.npz`, and `cross_section_comparison.csv` contain the numerical comparisons.
- `dss/dss_manuscript_addition.tex` is a draft manuscript insert with actual measured benchmark values.
- `test_dss_asn.py` checks target construction, disjoint splits, unmodified branch labels, checkpoint reloads, symbolic extraction, validation-based selection, and reproducibility of saved test metrics.

From the repository root, in an environment containing the repository dependencies plus the report dependencies:

```bash
python -m experiments.molecules.train --domain local
python -m experiments.molecules.train --domain broad
python -m experiments.molecules.dss_report
python -m pytest tests/molecules
```

The CPU training runs used PyTorch @@TORCH@@ with one computation thread per process. No repository model source was changed. The temporary execution environment is `/tmp/nmrkan-analysis-venv`. The HTML embeds its figures and uses native MathML; the notebook retains editable equations and links to the saved artifacts.
"""
    repl = {
        "POINT": center_table,
        "LOCAL_RMSE": f"{local['evaluations']['ASN']['frequency_rmse_hz']:.7g}",
        "LOCAL_MAX": f"{local['evaluations']['ASN']['frequency_max_abs_hz']:.7g}",
        "POINT_MAX": f"{max(abs(np.array(dss['error_hz']))):.7g}",
        "LOCAL_SEED": str(local["selected_seed"]),
        "LOCAL_EPOCH": str(local["selected_epoch"]),
        "ACTIVATIONS": support,
        "DEGREES": str(lf["degree"]),
        "EXTRACTION": f"{lf['max_extraction_error_dimensionless']:.3g}",
        "COEFFICIENTS": coef_table,
        "POLYNOMIALS": "\n\n".join(polynomial_lines),
        "SECTION_RMSE": f"{local['section_in_domain']['frequency_rmse_hz']:.7g}",
        "SECTION_MAX": f"{local['section_in_domain']['frequency_max_abs_hz']:.7g}",
        "DERIVATIVES": derivative_table,
        "LOCAL_TABLE": metric_table(local),
        "BROAD_TABLE": metric_table(broad),
        "SEEDS": seed_table,
        "BROAD_POINT": ", ".join(
            f"{v:.6f}" for v in broad["dss_point"]["asn_frequencies_hz"]
        ),
        "LOCAL_MEAN": f"{localstats['mean']:.6g}",
        "LOCAL_SD": f"{localstats['std']:.6g}",
        "BROAD_MEAN": f"{broadstats['mean']:.6g}",
        "BROAD_SD": f"{broadstats['std']:.6g}",
        "LOCAL_VIOLATIONS": str(local["evaluations"]["ASN"]["ordering_violations"]),
        "BROAD_VIOLATIONS": str(broad["evaluations"]["ASN"]["ordering_violations"]),
        "LOCAL_BOUNDARY": f"{local['boundary_challenge']['frequency_max_abs_hz']:.6g}",
        "BROAD_BOUNDARY": f"{broad['boundary_challenge']['frequency_max_abs_hz']:.6g}",
        "LOCAL_BOUNDARY_VIOLATIONS": str(
            local["boundary_challenge"]["ordering_violations"]
        ),
        "BROAD_BOUNDARY_VIOLATIONS": str(
            broad["boundary_challenge"]["ordering_violations"]
        ),
        "ROUNDED_POINT": ", ".join(
            f"{v:.6f}" for v in local["paper_dss_frequencies_hz"]
        ),
        "TORCH": local["config"]["torch"],
    }
    for key, value in repl.items():
        source = source.replace("@@" + key + "@@", value)
    if "@@" in source:
        raise ValueError("Unfilled report placeholder")
    return source


def write_report(
    source,
    folder_root=FOLDER,
    stem="dss_asn_comparison",
    title="DSS learned ASN versus analytical solution",
):
    FOLDER = folder_root
    dirname = FOLDER.name
    equations = []

    def math_replace(match, display):
        equations.append(
            convert(match.group(1), display="block" if display else "inline")
        )
        token = f"MATHTOKEN{len(equations) - 1:05d}END"
        return "\n\n" + token + "\n\n" if display else token

    body = re.sub(r"\$\$(.*?)\$\$", lambda m: math_replace(m, True), source, flags=re.S)
    body = re.sub(r"\$([^$\n]+)\$", lambda m: math_replace(m, False), body)
    body = markdown.markdown(body, extensions=["tables", "fenced_code"])
    for i, equation in enumerate(equations):
        body = body.replace(f"MATHTOKEN{i:05d}END", equation)
    for path in FOLDER.glob("*.png"):
        body = body.replace(
            f'src="{dirname}/{path.name}"',
            'src="data:image/png;base64,'
            + base64.b64encode(path.read_bytes()).decode()
            + '"',
        )
    style = """body{max-width:1050px;margin:3em auto;padding:0 2em;font:17px/1.55 Georgia,serif;color:#1b2838}h1,h2{font-family:Arial,sans-serif;color:#173c55;line-height:1.25}h2{margin-top:2em}a{color:#126994}table{border-collapse:collapse;width:100%;font:13px/1.4 Arial,sans-serif;margin:1.3em 0}th,td{padding:.5em;border-bottom:1px solid #d7e2e8;text-align:left}th{background:#edf4f7}img{width:100%;height:auto}pre{font-size:12px;background:#edf4f7;padding:1em;overflow:auto}code{font-size:.85em}math[display=block]{display:block;overflow:auto;padding:.5em 0}@media print{body{font-size:10pt;margin:0;padding:0}table{font-size:8pt}h2{break-after:avoid}img,table{break-inside:avoid}math[display=block]{font-size:.94em;overflow:visible}@page{size:A4;margin:15mm}}"""
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'
        + title
        + "</title><style>"
        + style
        + "</style></head><body>"
        + body
        + "</body></html>"
    )
    (ROOT / (stem + ".html")).write_text(html)
    cells = []
    for part in re.split(r"(?=^## )", source, flags=re.M):
        attachments = {}
        for match in list(
            re.finditer(r"!\[([^]]*)\]\(" + re.escape(dirname) + r"/([^)]*)\)", part)
        ):
            name = match.group(2)
            attachments[name] = {
                "image/png": base64.b64encode((FOLDER / name).read_bytes()).decode()
            }
            part = part.replace(
                match.group(0), f"![{match.group(1)}](attachment:{name})"
            )
        cell = dict(
            cell_type="markdown",
            id=f"asn-comparison-{len(cells):03d}",
            metadata={},
            source=part.strip().splitlines(keepends=True),
        )
        if attachments:
            cell["attachments"] = attachments
        cells.append(cell)
    example = "from pathlib import Path\nimport json, sympy as sp, numpy as np\nfrom experiments.molecules.paths import OUTPUT_ROOT as root\nf = json.loads((root/'MOLECULE/local/learned_formulas.json').read_text())\nsymbols=sp.symbols('xA xX u v')\nexpressions=[sp.sympify(e) for e in f['formulas']]\nfor i,e in enumerate(expressions):\n    print('lambda',i,'=',e)\n"
    example = example.replace("MOLECULE", dirname)
    cells.append(
        dict(
            cell_type="code",
            id="load-learned-formulas",
            metadata={},
            execution_count=None,
            outputs=[],
            source=example.splitlines(keepends=True),
        )
    )
    notebook = dict(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        nbformat=4,
        nbformat_minor=5,
    )
    (ROOT / (stem + ".ipynb")).write_text(json.dumps(notebook, indent=1) + "\n")


def manuscript(results):
    r = results["local"]
    d = r["dss_point"]
    evals = r["evaluations_extended"]
    text = r"""% Draft insertion: generated from completed training artifacts, not hypothetical results.
\subsection{A molecular application: DSS}
We extended the activation-selection network to a three-methylene chain with
three distinct geminal couplings. For DSS, the coupling vector in Hz was
$(-13.758727,-13.408159,-14.121958,5.269170,6.597217)$, where the last two
entries are full nearest-neighbour coupling differences. The four-state
secular parity block retains type-II mixing. With $s=-J_M>0$ and
$E_{\rm ref}=(-3J_A+J_M-3J_X)/4$, its four dimensionless inputs are
$(x_A,x_X,u,v)=((J_A-J_M)/s,(J_X-J_M)/s,d/s,e/s)$, and
$E_i=E_{\rm ref}+s\lambda_i$. Numerical targets were ordered as
$\lambda_0\le\lambda_1\le\lambda_2\le\lambda_3$.

The molecular application used a (4,16,4) ASN with the activation library
$\{0,x,x^2\}$. We independently varied each geminal coupling by $\pm0.3$ Hz
and each coupling difference by $\pm20\%$ around the DSS reference. These
ranges define a numerical approximation domain, not experimental confidence
intervals. Of 100000 samples, 80000 were used for training, 10000 for
validation, and 10000 for testing. Training-set affine input and output
standardization was applied and algebraically undone during extraction.
Adam training used 1000 epochs, batch size 4096, learning rate $10^{-3}$,
and MAE on standardized targets. Softmax with a $0.1L_{0.5}$ penalty was
used for 900 epochs, followed by 100 epochs of sparsemax without the penalty.
Five initialization seeds were evaluated; checkpoint and representative-seed
selection used validation eigenvalue RMSE only.

An independent exact cubic construction supplies the analytical reference.
Let the positive normal-mode energies satisfy $\omega_1\le\omega_2\le\omega_3$.
The three principal frequencies are
$f_{12}=\omega_3-\omega_2$, $f_{23}=\omega_2-\omega_1$, and
$f_{13}=f_{12}+f_{23}$. Learned frequencies were formed from the corresponding
fixed eigenvalue differences, without sorting the network outputs afterward.

@@RESULTS@@

On the section $d=5.269170t$, $e=6.597217t$, with the DSS geminals fixed,
substitution into the trained four-input network gives
@@FORMULAS@@
These expressions were extracted by composing the trained layers; no
additional one-dimensional fit was performed. They approximate the
finite-domain dependence and are not perturbative Taylor coefficients.
The spectral comparison uses common illustrative preparation/readout
amplitudes and a 0.1 Hz Lorentzian linewidth, replacing only the line centers
with learned predictions. It therefore tests frequency prediction rather
than a learned model of experimental intensities.

The polynomial least-squares baseline must be retained in interpreting
these results: compact learned formulas do not establish an intrinsic
accuracy advantage over an appropriate fixed basis. A separate broader
parameter-domain run and decoupled-boundary checks are reported in the
supplementary comparison. No claim of uniform accuracy or guaranteed
ordering outside the demonstrated molecular domain is made.
"""
    result = (
        f"The selected ASN achieved a held-out frequency RMSE of {evals['ASN']['frequency_rmse_hz']:.6g} Hz "
        f"and a maximum line error of {evals['ASN']['frequency_max_abs_hz']:.6g} Hz. "
        f"The quadratic and quartic least-squares baselines gave frequency RMSEs of "
        f"{evals['OLS degree 2']['frequency_rmse_hz']:.6g} and {evals['OLS degree 4']['frequency_rmse_hz']:.6g} Hz, respectively. "
        "At the DSS reference, the exact frequencies were "
        + ", ".join(f"{v:.6f}" for v in d["exact_frequencies_hz"])
        + " Hz, while the learned frequencies were "
        + ", ".join(f"{v:.6f}" for v in d["asn_frequencies_hz"])
        + " Hz. "
        f"The selected model had {evals['ASN']['ordering_violations']} ordering violations on the 10000-point local test set. "
        f"Three-decimal coefficient rounding gave a local frequency RMSE of {evals['ASN paper-style 3-decimal rounding']['frequency_rmse_hz']:.6g} Hz."
    )
    t = sp.Symbol("t")
    formula = []
    for label, coeff in zip(
        (12, 23, 13), r["formulas"]["cross_section_coefficients_hz"]
    ):
        expr = sum(
            sp.Float(round(float(c), 9)) * t**k
            for k, c in enumerate(coeff)
            if abs(c) > 1e-10
        )
        formula.append(
            "\\[\\widehat f_{"
            + str(label)
            + "}(t)="
            + sp.latex(expr)
            + "\\quad\\mathrm{Hz}.\\]"
        )
    text = text.replace("@@RESULTS@@", result).replace(
        "@@FORMULAS@@", "\n".join(formula)
    )
    (FOLDER / "dss_manuscript_addition.tex").write_text(text)


def main():
    results = {domain: analyze(domain) for domain in ("local", "broad")}
    figures(results)
    write_report(build_text(results))
    manuscript(results)
    for domain, r in results.items():
        print(
            domain,
            "selected seed",
            r["selected_seed"],
            "ASN",
            r["evaluations"]["ASN"],
            "DSS",
            r["dss_point"],
            flush=True,
        )
    print("Reports, figures, and manuscript insertion written.")


if __name__ == "__main__":
    main()
