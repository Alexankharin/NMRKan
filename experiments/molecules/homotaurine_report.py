"""Homotaurine analytical derivation, actual trained ASN comparison, and paper insert.

Run both train_dss_asn.py --molecule homotaurine domains before this script.
Uses the shared, independently tested molecular solver and report utilities.
"""

import json
import math

import numpy as np
import sympy as sp

from nmrkan.physics.three_ch2 import EXAMPLES, hamiltonian, normalize_parameters
from nmrkan.physics.real_molecules import (
    analytical_eigenvalues,
    analytical_frequencies,
    normal_modes,
    cross_section_taylor,
    effective_three_state,
    principal_gaps,
    singlet_observable,
    spectral_lines,
    lorentzian_spectrum,
    make_figures,
)
from experiments.molecules.train import (
    HOMOTAURINE as P,
    write_rows,
)
from experiments.molecules.dss_report import (
    analyze,
    figures,
    metric_table,
    write_report,
)

from .paths import OUTPUT_ROOT as ROOT

OUT = ROOT / "homotaurine"
Q = list(EXAMPLES.values())[1]["dJAX"]


def analytical_artifacts():
    OUT.mkdir(exist_ok=True)
    x, s, r = normalize_parameters(*P)
    E = analytical_eigenvalues(P)
    f = analytical_frequencies(P)
    H = hamiltonian(*P)
    mean = P.copy()
    mean[:3] = P[:3].mean()
    variants = {
        "Exact nearest-neighbor": f,
        "Drop type-II (bare 3x3)": principal_gaps(np.linalg.eigvalsh(H[:3, :3])),
        "Second-order effective 3x3": principal_gaps(
            np.linalg.eigvalsh(effective_three_state(P))
        ),
        "Replace geminals by their mean": analytical_frequencies(mean),
        "Fitted q, even parity": principal_gaps(
            np.linalg.eigvalsh(hamiltonian(*P, dJAX=Q))[1:]
        ),
        "Fitted q, odd parity (matched)": principal_gaps(
            np.linalg.eigvalsh(hamiltonian(*P, dJAX=Q, parity="odd"))[:3]
        )[[1, 0, 2]],
    }
    rows = []
    for t in np.linspace(0, 1.5, 601):
        p = P.copy()
        p[3:] *= t
        e = analytical_eigenvalues(p)
        row = dict(t=t, yAM=t * x[2], yMX=t * x[3])
        row.update({f"E{i}_hz": v for i, v in enumerate(e)})
        row.update({f"lambda{i}": v for i, v in enumerate((e - r) / s)})
        for prefix, values in (
            ("f", analytical_frequencies(p)),
            ("bare_f", principal_gaps(np.linalg.eigvalsh(hamiltonian(*p)[:3, :3]))),
            (
                "effective_f",
                principal_gaps(np.linalg.eigvalsh(effective_three_state(p))),
            ),
        ):
            row.update(
                {f"{prefix}{label}_hz": v for label, v in zip((12, 23, 13), values)}
            )
        rows.append(row)
    write_rows(OUT / "homotaurine_cross_section.csv", rows)
    a, b, c, d, e = P / s
    coefficients = [
        [a * a + b * b + c * c, d * d + e * e],
        [
            a * a * b * b + a * a * c * c + b * b * c * c,
            a * a * e * e + c * c * d * d,
            d * d * e * e,
        ],
        [(a * b * c) ** 2],
    ]
    taylor = cross_section_taylor(P)
    error = []
    for t in np.linspace(0.9, 1.1, 401):
        p = P.copy()
        p[3:] *= t
        error.append(
            taylor @ np.array([1, t - 1, (t - 1) ** 2]) - analytical_frequencies(p)
        )
    # Analytic first derivative with respect to q (Hellmann--Feynman).
    Qmatrix = hamiltonian(0, 0, 0, 0, 0, dJAX=1)
    _, U = np.linalg.eigh(H)
    qderivative = np.diag(U.T @ Qmatrix @ U)
    qgapderivative = np.array(
        [
            qderivative[2] - qderivative[1],
            qderivative[3] - qderivative[2],
            qderivative[3] - qderivative[1],
        ]
    )
    all_lines = []
    spectrum_axis = np.linspace(0, 36, 18001)
    spectra = {"frequency_hz": spectrum_axis}
    for q in (0.0, Q):
        for parity in ("even", "odd"):
            O = singlet_observable("A", parity)
            _, lines = spectral_lines(hamiltonian(*P, dJAX=q, parity=parity), O, O)
            all_lines.extend(dict(q_hz=q, parity=parity, **line) for line in lines)
            spectra[f"q{q:g}_{parity}"] = lorentzian_spectrum(spectrum_axis, lines)
    write_rows(OUT / "illustrative_transition_amplitudes.csv", all_lines)
    write_rows(
        OUT / "homotaurine_illustrative_spectrum.csv",
        [{k: v[i] for k, v in spectra.items()} for i in range(len(spectrum_axis))],
    )
    result = dict(
        parameters_hz=P.tolist(),
        q_hz=Q,
        features=x.tolist(),
        scale_hz=s,
        reference_hz=r,
        provenance=list(EXAMPLES)[1],
        eigenvalues_hz=E.tolist(),
        normalized_eigenvalues=((E - r) / s).tolist(),
        modes_hz=normal_modes(P).tolist(),
        frequencies_hz=f.tolist(),
        variants={k: v.tolist() for k, v in variants.items()},
        approximation_max_errors_hz={
            k: float(max(abs(v - f))) for k, v in variants.items()
        },
        cubic_section_coefficients_even_powers=coefficients,
        taylor_about_t1_hz=taylor.tolist(),
        taylor_max_error_0p9_to_1p1_hz=float(np.max(abs(np.array(error)))),
        q_gap_derivative=qgapderivative.tolist(),
        q_first_order_even_frequencies_hz=(f + Q * qgapderivative).tolist(),
        q_first_order_max_error_hz=float(
            np.max(abs(f + Q * qgapderivative - variants["Fitted q, even parity"]))
        ),
        parity_max_splitting_hz=float(
            max(
                abs(
                    variants["Fitted q, even parity"]
                    - variants["Fitted q, odd parity (matched)"]
                )
            )
        ),
        principal_TTT_weights=(U[3, 1:] ** 2).tolist(),
    )
    write_rows(
        OUT / "model_comparison.csv",
        [
            dict(
                model=k,
                f12_hz=v[0],
                f23_hz=v[1],
                f13_hz=v[2],
                max_error_hz=max(abs(v - f)),
            )
            for k, v in variants.items()
        ],
    )
    (OUT / "analytical_results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_figures(OUT, P, rows, result, name="Homotaurine")
    return result


def polynomial_lines(coefficients, variable="t", prefix=r"\widehat f"):
    t = sp.Symbol(variable)
    lines = []
    for label, coeff in zip((12, 23, 13), coefficients):
        expr = sum(
            sp.Float(round(float(c), 9)) * t**k
            for k, c in enumerate(coeff)
            if abs(c) > 1e-12
        )
        lines.append(
            "$$"
            + prefix
            + "_{"
            + str(label)
            + "}="
            + sp.latex(expr)
            + r"\quad\mathrm{Hz}.$$"
        )
    return "\n\n".join(lines)


def build_text(physics, results):
    local, broad = results["local"], results["broad"]
    point = local["reference_point"]
    ev = local["evaluations_extended"]
    pointtable = "| Line | Exact q=0 / Hz | Trained ASN / Hz | Error / mHz | Exact fitted q, even / Hz |\n|---|---:|---:|---:|---:|\n"
    for k, label in enumerate((12, 23, 13)):
        pointtable += f"| f{label} | {point['exact_frequencies_hz'][k]:.9f} | {point['asn_frequencies_hz'][k]:.9f} | {1000 * point['error_hz'][k]:+.6f} | {physics['variants']['Fitted q, even parity'][k]:.9f} |\n"
    approximation = "| Model | f12 / Hz | f23 / Hz | f13 / Hz | Max. shift from q=0 / Hz |\n|---|---:|---:|---:|---:|\n"
    for key, vals in physics["variants"].items():
        approximation += (
            f"| {key} | "
            + " | ".join(f"{v:.9f}" for v in vals)
            + f" | {physics['approximation_max_errors_hz'][key]:.9f} |\n"
        )
    normalization = "(" + ", ".join(f"{v:.12f}" for v in physics["features"]) + ")"
    symbols = sp.symbols("xA xX u v")
    polys = [sp.Poly(sp.sympify(e), *symbols) for e in local["paper_formulas"]]
    powers = sorted(
        set(p for poly in polys for p, _ in poly.terms()), key=lambda p: (sum(p), p)
    )
    coefficients = "| Monomial | λ0 | λ1 | λ2 | λ3 |\n|---|---:|---:|---:|---:|\n"
    for power in powers:
        term = sp.prod(v**k for v, k in zip(symbols, power))
        coefficients += (
            "| $"
            + sp.latex(term)
            + "$ | "
            + " | ".join(f"{float(poly.coeff_monomial(term)):.3f}" for poly in polys)
            + " |\n"
        )
    seeds = "| Domain | Seed | Selected epoch | Validation energy RMSE (dimensionless) | Test frequency RMSE / Hz |\n|---|---:|---:|---:|---:|\n"
    for domain, result in results.items():
        for seed in result["seeds"]:
            mark = " (selected)" if seed["seed"] == result["selected_seed"] else ""
            seeds += f"| {domain} | {seed['seed']}{mark} | {seed['best_epoch']} | {seed['validation_rmse_dimensionless']:.7g} | {seed['test']['frequency_rmse_hz']:.7g} |\n"
    deriv = "| Line | Exact slope | ASN slope | Exact quadratic coefficient | ASN quadratic coefficient |\n|---|---:|---:|---:|---:|\n"
    for label, taylor, coeff in zip(
        (12, 23, 13),
        physics["taylor_about_t1_hz"],
        local["formulas"]["cross_section_coefficients_hz"],
    ):
        learned = [
            sum(c * math.comb(n, j) for n, c in enumerate(coeff) if n >= j)
            for j in range(3)
        ]
        deriv += f"| f{label} | {taylor[1]:.9f} | {learned[1]:.9f} | {taylor[2]:.9f} | {learned[2]:.9f} |\n"
    text = r"""
# Homotaurine: analytical solution and formulas learned by ASN

Real-molecule application for the revised manuscript · 18 October 2026

**Both results are provided.** The exact analytical construction is the reference; the other formulas are extracted from an actually trained `nmrkan.models.KharKAN`, the activation-selection network (ASN) used in the manuscript. Supervised targets come from numerical 4×4 diagonalization, not from the analytical cubic. This repeats the DSS protocol with the supplied homotaurine reference and a new local training distribution.

@@POINT@@

The selected local model has held-out frequency RMSE **@@RMSE@@ Hz**, maximum line error **@@MAX@@ Hz**, and **@@VIOLATIONS@@** ordering violations on 10,000 test points. These are surrogate errors against the nearest-neighbor secular model. They are not residuals against measured peaks. The last table column shows a separate physical correction, which the four-input q=0 ASN was not trained to predict.

![Homotaurine exact and trained predictions](homotaurine/homotaurine_asn_vs_analytical.png)

## 1. Molecular parameters and the experimental connection

The supplied `ASN_DSS_homotaurine_Hamiltonian.pdf` and `asn_three_ch2.py` specify

$$ (J_A,J_M,J_X,d,e)=(-12.743923,-13.969277,-14.109177,3.093406,2.344390)\ \mathrm{Hz},$$
$$q=J_{AX}-J_{AX'}=0.027306\ \mathrm{Hz}.$$

Here d and e are full coupling differences; off-diagonal matrix elements are d/2 and e/2. The source attributes this row to a 15 August 2026 email. It explicitly flags a filename labeled 298 K but a body labeled 289 K, and an unreconciled subsequent revision. We retain the internally consistent physical row as an illustrative molecular reference. Its temperature and final-fit status remain unresolved. The raw spectrum and source emails are not available here.

The experimental precedent is [Sheberstov, Sonnefeld and Bodenhausen, J. Chem. Phys. 160, 144308 (2024)](https://doi.org/10.1063/5.0196808). The [authors' accepted manuscript, Fig. 9b,d and Experimental section](https://www.researchgate.net/publication/379772660_Collective_long-lived_zero-quantum_coherences_in_aliphatic_chains) describes homotaurine ZQ sections with in-phase DQ-AM-SLIC preparation and SQ-A-SLIC or SQ-X-SLIC reconversion. It reports measurements at 298 K and 500 MHz on a D₂O mixture containing approximately 50 mM of each molecule. This establishes the relevant experiment class; it does not resolve the temperature of the supplied 2026 parameter row. The published ZQ linewidth scale is about 0.1 Hz, which we use for illustrative Lorentzians. No experimental intensities or fit uncertainties are inferred from it.

## 2. Hamiltonian and what can be neglected

For $S_J=J_A+J_M+J_X$, the even block in the basis $(TSS,STS,SST,TTT)$ is

$$H_+=-\frac{3S_J}{4}I+\begin{pmatrix}
J_A&d/2&q/2&e/2\\d/2&J_M&e/2&q/2\\q/2&e/2&J_X&d/2\\e/2&q/2&d/2&S_J
\end{pmatrix}.$$

Hamiltonians are in Hz and time evolution is $\exp(-i2\pi H\tau)$. The common pair chemical-shift and mean interpair secular terms vanish on the product singlet/central-triplet subspace because $L_{i,z}=I_{i,z}+I_{i',z}$ annihilates both states. A scalar identity term can be subtracted. Interpair nonsecular flip-flops are a separate approximation: their corrections require chemical-shift separations and individual couplings absent from the five-parameter row. The model assumes chemically equivalent protons within each pair, isotropic intrapair coupling, secular interpair coupling, and no RF during evolution.

@@APPROXIMATIONS@@

**Keep unequal geminals and type-II mixing.** Averaging the geminals shifts a principal line by 0.278275 Hz. Dropping the fourth state shifts a line by 0.053710 Hz. Three observed principal peaks do not justify removing that state: weakly observed states still shift energies by virtual mixing. At this reference the fourth-basis-state probabilities in the three principal eigenstates are @@WEIGHTS@@ percent.

A controlled elimination writes $H=\begin{pmatrix}A&w\\w^T&b\end{pmatrix}$. The exact energy-dependent reduced equation is $(A+ww^T/(E-b))\psi=E\psi$. In an eigenbasis of A, a Hermitian second-order approximation is

$$ (H_{\rm eff})_{ij}=a_i\delta_{ij}+\frac{w_iw_j}{2}\left[(a_i-b)^{-1}+(a_j-b)^{-1}\right].$$

Its maximum principal-line error here is 0.000287194 Hz. It is much better than simply deleting the fourth state; keep its induced off-diagonal elements. The reference cubic below is exact within the q=0 block.

**The q correction matters for precision.** The supplied q changes a single-parity line by up to 0.018095 Hz and separates matched even/odd lines by up to **@@PARITY_SPLIT@@ Hz**. They may overlap at a 0.1 Hz linewidth, but q should be included for millihertz line-center comparisons. The two parity blocks cannot automatically be represented by a single triplet when that splitting is resolved. Their relative preparation/readout weights are experiment-dependent. The first-order shifts have opposite signs in the matched parity blocks: an equally weighted pair of lines has a centroid shift only at second order. Thus the quoted single-parity shift is not automatically a shift of the maximum of an unresolved experimental peak; q can also alter its shape.

An analytical first-order correction is $E_i(q)=E_i(0)+q\langle i|Q|i\rangle+O(q^2)$, with $Q=\partial H/\partial q$ containing 1/2 at the four q positions. The principal-frequency derivatives at q=0 are **@@Q_DERIV@@** (Hz per Hz). At the supplied q the first-order even-block frequencies are **@@Q_FIRST@@ Hz**, with maximum error **@@Q_ERROR@@ Hz** against full 4×4 diagonalization. This is a physics correction, not an extra learned output. A model that varies q independently needs a fifth normalized input.

![All transitions and the fitted q correction](homotaurine/homotaurine_observability.png)

## 3. Exact normalization, analytical eigenvalues, and branch labels

Use the same normalization as for DSS:

$$s=-J_M>0,\quad E_{\rm ref}=(-3J_A+J_M-3J_X)/4,$$
$$x_A=(J_A-J_M)/s,\quad x_X=(J_X-J_M)/s,\quad u=d/s,\quad v=e/s.$$

For homotaurine, $s=13.969277$ Hz, $E_{\rm ref}=16.64750575$ Hz, and $(x_A,x_X,u,v)=$ **@@FEATURES@@**. Five dimensional parameters thus become four independent inputs. For q=0,

$$K=\frac{H-E_{\rm ref}I}{s}=\begin{pmatrix}
x_A&u/2&0&v/2\\u/2&0&v/2&0\\0&v/2&x_X&u/2\\v/2&0&u/2&-2+x_A+x_X
\end{pmatrix}.$$

Let $a=J_A/s=x_A-1$, $c=J_X/s=x_X-1$, and

$$B=\begin{pmatrix}a&u&0\\0&-1&v\\0&0&c\end{pmatrix}.$$

The three ordered squared singular values of B are the roots $z_1\le z_2\le z_3$ of

$$z^3-C_1z^2+C_2z-C_3=0,$$
$$C_1=a^2+1+c^2+u^2+v^2,$$
$$C_2=a^2+a^2c^2+c^2+a^2v^2+c^2u^2+u^2v^2,\qquad C_3=a^2c^2.$$

For an explicit trigonometric expression, set $m=C_1/3$, $r=\sqrt{C_1^2-3C_2}/3$, and

$$\theta=\frac13\arccos\!\left(\frac{2C_1^3-9C_1C_2+27C_3}{2(C_1^2-3C_2)^{3/2}}\right).$$
$$ (z_1,z_2,z_3)=m+2r\left(\cos(\theta+2\pi/3),\cos(\theta-2\pi/3),\cos\theta\right).$$

At r=0 all roots equal m. Numerical evaluation uses centered matrix invariants, with a symmetric eigensolver fallback very near repeated roots to avoid cancellation. This does not change the analytical identity. Define $\omega_i=s\sqrt{z_i}$ and $c_0=-S_J/4$. For negative geminals,

$$E_0=c_0-(\omega_1+\omega_2+\omega_3)/2,$$
$$E_1=c_0+(\omega_1+\omega_2-\omega_3)/2,$$
$$E_2=c_0+(\omega_1-\omega_2+\omega_3)/2,$$
$$E_3=c_0+(-\omega_1+\omega_2+\omega_3)/2.$$

These are ordered physical energies $E_0\le E_1\le E_2\le E_3$. The supplied point has eigenvalues **@@ENERGIES@@ Hz**, and normal-mode energies **@@MODES@@ Hz**. The principal transitions are

$$f_{12}=E_2-E_1=\omega_3-\omega_2,\quad f_{23}=E_3-E_2=\omega_2-\omega_1,$$
$$f_{13}=E_3-E_1=f_{12}+f_{23}.$$

Keep these index labels even if the two adjacent frequencies cross. Sorting their magnitudes would swap their identities. Ordered energy surfaces are continuous but may have cusps at degeneracy; a fixed ordered label is not a globally smooth eigenvector identity. Since s is positive, normalized energies have the same ascending order. All four targets are retained regardless of their sign.

![Homotaurine eigenvalue surfaces](homotaurine/homotaurine_eigenvalue_surfaces.png)

## 4. Analytical and learned cross-sections of the same model

Fix the supplied geminals and put $d(t)=3.093406t$ Hz and $e(t)=2.344390t$ Hz. The physical reference is t=1; this is a mathematical parameter section, not a temperature calibration. Along it,

$$C_1(t)=2.852388810881+0.077202373264t^2,$$
$$C_2(t)=2.701400997954+0.073465038627t^2+0.001381140469t^4,$$
$$C_3(t)=0.849012187073.$$

Insert these expressions into the ordered root formulas to obtain the exact analytical spectral section. Differentiating that solution at $\delta=t-1=0$ gives the following Taylor polynomials, with omitted terms $O(\delta^3)$:

@@TAYLOR@@

Their sampled maximum error on $0.9\le t\le1.1$ is **@@TAYLOR_ERROR@@ Hz**. The derivatives are calculated analytically with Hermitian eigenvalue derivative identities; these are not fitted polynomials.

Substitution of this section into the **actual trained four-input ASN** gives:

@@LEARNED@@

These expressions are composed from learned layer weights and activation mixtures. No one-dimensional fitting was performed. Full coefficients are saved; nine decimals are shown here. They approximate a finite training domain and should not be identified with the exact Taylor coefficients. On the local 161-point section $0.8\le t\le1.2$, their frequency RMSE is **@@SECTION_RMSE@@ Hz** and maximum error **@@SECTION_MAX@@ Hz**. The derivative comparison about t=1 is:

@@DERIVATIVES@@

![Exact cross-section and reduced-model errors](homotaurine/homotaurine_cross_section.png)

## 5. Actual training and extracted four-input formulas

The repository class `nmrkan.models.KharKAN` implements the manuscript's ASN with activation choices {0,x,x²}; this is not a spline-edge KAN. We trained (4,16,4) networks with biases using Adam, learning rate 0.001, batch size 4096, and 1000 epochs. Epochs 1–900 use softmax weights and a 0.1×L0.5 regularizer; epochs 901–1000 use sparsemax without the regularizer. The loss is MAE on training-standardized targets. Affine input and output conditioning is fitted only to training data and undone algebraically in the extracted formulas. This conditioning is an explicit extension of the original manuscript's protocol, matching the DSS experiment.

The **local domain** independently varies each homotaurine geminal by ±0.3 Hz and each coupling difference by ±20%. The **broad domain** uses the same original box as DSS: geminals in [−15,−12.5] Hz and differences in [0.1,7] Hz. The local J_A upper bound is −12.443923 Hz, so the local domain is not wholly contained in the broad box. These are numerical approximation domains, not parameter confidence intervals.

Each domain uses 100,000 vectors generated with NumPy seed 42, split into 80,000 training, 10,000 validation, and 10,000 test vectors. Five initialization/minibatch seeds (42–46) are trained. Checkpoint and seed selection minimize unstandardized dimensionless eigenvalue RMSE on validation samples, restricted to the sparsemax phase. Test evaluation follows selection. The exact reference vector and section were not deliberately inserted in training. The broad run uses the same physical distribution and seeds as DSS: it is the same general Hamiltonian benchmark evaluated on a different molecular section, not an independent second-molecule training population.

The selected local model is seed **@@SEED@@**, epoch **@@EPOCH@@**. Extracted degrees are **@@DEGREES@@**. Numerical targets come only from `numpy.linalg.eigvalsh` on the physical 4×4 block, followed by normalization. The model never receives cubic invariants, singular modes, or analytical derivatives. We impose no post-prediction sorting, positivity repair, symmetry averaging, or trace projection.

The following coefficient table defines $\widehat\lambda_i=\sum_m c_{im}m(x_A,x_X,u,v)$ after the manuscript's presentation step: discard coefficients with magnitude below 0.001 and round remaining coefficients to three decimals. The main performance results use the separate full-precision formulas.

@@COEFFICIENTS@@

Full expressions are in `homotaurine/local/learned_formulas.json` and `.tex`. Their maximum discrepancy from the double-precision trained network on 1000 validation points is **@@EXTRACTION@@**. Checkpoint reload and formula agreement are also checked on held-out samples. The three-decimal formulas yield **@@ROUNDED_POINT@@ Hz** at the reference; their distinct error row below quantifies the cost of rounding.

## 6. Held-out benchmarks and branch-order checks

Each method within a domain uses the same test samples. Complete quadratic (15 monomials) and quartic (70 monomials) ordinary least-squares fits use the same 80,000 training samples. Their squared-error objective differs from the ASN's MAE/regularization objective. The analytical cubic, effective 3×3, and bare 3×3 are physics references with no fitted coefficients.

**Homotaurine local domain**

@@LOCAL_TABLE@@

**Broad supplied domain**

@@BROAD_TABLE@@

The broad model predicts **@@BROAD_POINT@@ Hz** at homotaurine. Its global test error must not be confused with local approximation accuracy. Seed variability is shown below; all seeds share the same dataset, so this measures optimization variation rather than experimental uncertainty.

@@SEEDS@@

The selected local and broad models have **@@VIOLATIONS@@** and **@@BROAD_VIOLATIONS@@** ordering violations, respectively, on their 10,000 test points. A separate 2000-point challenge with broad-box geminals sets d=0 or e=0; these are outside both sampled boxes and probe the decoupled boundary. Its local/broad maximum frequency errors are **@@BOUNDARY_ERRORS@@ Hz**, with **@@BOUNDARY_VIOLATIONS@@** violations. We do not sort predictions afterward to hide these failures. Zero sampled violations would still not prove global ordering.

The learned identity $\widehat f_{13}=\widehat f_{12}+\widehat f_{23}$ holds by construction. Trace errors, per-seed statistics, boundary metrics and all configurations are retained in JSON. Broad-domain ordering cusps are an intrinsic limitation of a finite smooth polynomial representation.

![Homotaurine local and broad comparisons](homotaurine/homotaurine_asn_domain_comparison.png)

![Actual training histories](homotaurine/homotaurine_asn_training.png)

## 7. From three energy gaps to an observable spectrum

For preparation $\rho$ and readout O, the signal is $S(\tau)=\mathrm{Tr}[Oe^{-i2\pi H\tau}\rho e^{i2\pi H\tau}]$. Each energy pair contributes a cosine amplitude $2\mathrm{Re}(\rho_{ij}O_{ji})$ and sine amplitude $-2\mathrm{Im}(\rho_{ij}O_{ji})$. A frequency need not be visible merely because an energy difference exists.

For a reproducible illustration we use the local-A singlet deviation operator $\rho=O=P_{S,A}-I/2=\mathrm{diag}(-1/2,1/2,1/2,-1/2)$ in the even block. It describes a deviation from background, not a positive density operator by itself. At q=0 the three principal cosine amplitudes in f12,f23,f13 order are 0.084349, 0.322096, and 0.169245. Three additional weak transitions occur near 25.3591, 27.3270, and 29.5124 Hz. They are not forbidden; their illustrative amplitudes are 0.002392, 0.003910, and 0.000375.

The main comparison plots use the same three principal amplitudes and 0.1 Hz Lorentzian FWHM for both methods, replacing only the frequencies with ASN predictions. The separate full-spectrum CSV retains all six lines and both parity blocks for q=0 and fitted q. Equal parity weighting in the q illustration is an assumption; actual poly-SLIC preparation and reconversion must determine it. The ASN learns eigenvalues, not eigenvectors or pulse efficiencies. It does not predict experimental intensities or relaxation.

## 8. Paper narrative and reproducibility

The molecular demonstration now follows the full sequence requested by the editor: supplied real-molecule couplings, justified reduced Hamiltonian, normalization, numerical supervised targets, actual ASN training, extraction of explicit formulas, and comparison of observable line centers with an independent analytical solution. Retain the polynomial and effective-Hamiltonian baselines when interpreting accuracy. The results establish a useful learned approximation over a stated domain; they do not establish superiority over an appropriate polynomial basis or an exact small-matrix solver.

Homotaurine also exposes a physical-model limitation: a small surrogate error does not remove the omitted q correction or resolve the supplied fit's temperature ambiguity. The analytical and learned q=0 results are valid computational benchmarks. Experimental validation requires the final physical parameter row and measured spectrum, together with the preparation/readout model. The editor's separate request for a contemporary symbolic-regression baseline such as PySR/EQL remains separate; none is claimed here. Any manuscript statement that AI was not used to produce research calculations should be reconciled with the use of these artifacts.

- `homotaurine/homotaurine_manuscript_addition.tex`: manuscript insertion with measured benchmark results.
- `homotaurine/analytical_results.json`: exact energies, approximations, q correction, and Taylor coefficients.
- `homotaurine/{local,broad}/`: datasets, split indices, configurations, hashes, five checkpoints per domain, training histories, full/rounded formulas, held-out predictions and section CSVs.
- Figures are saved as PDF, SVG and PNG; this HTML and the notebook embed images for portability.

Reproduce from the repository root:

```bash
python -m experiments.molecules.train --molecule homotaurine --domain local
python -m experiments.molecules.train --molecule homotaurine --domain broad
python -m experiments.molecules.homotaurine_report
python -m pytest tests/molecules
```

CPU training uses one computation thread per process; dependency versions and source/data hashes are recorded in each config. No repository model source was changed. Full-precision machine-readable artifacts should be used for numerical evaluation; printed coefficients are rounded.
"""
    fmt = lambda arr: ", ".join(f"{v:.9f}" for v in arr)
    repl = dict(
        POINT=pointtable,
        RMSE=f"{ev['ASN']['frequency_rmse_hz']:.7g}",
        MAX=f"{ev['ASN']['frequency_max_abs_hz']:.7g}",
        VIOLATIONS=str(ev["ASN"]["ordering_violations"]),
        APPROXIMATIONS=approximation,
        WEIGHTS=fmt(100 * np.array(physics["principal_TTT_weights"])),
        PARITY_SPLIT=f"{physics['parity_max_splitting_hz']:.9f}",
        Q_DERIV=fmt(physics["q_gap_derivative"]),
        Q_FIRST=fmt(physics["q_first_order_even_frequencies_hz"]),
        Q_ERROR=f"{physics['q_first_order_max_error_hz']:.7g}",
        FEATURES=normalization,
        ENERGIES=fmt(physics["eigenvalues_hz"]),
        MODES=fmt(physics["modes_hz"]),
        TAYLOR=polynomial_lines(physics["taylor_about_t1_hz"], "delta", "f"),
        TAYLOR_ERROR=f"{physics['taylor_max_error_0p9_to_1p1_hz']:.7g}",
        LEARNED=polynomial_lines(local["formulas"]["cross_section_coefficients_hz"]),
        SECTION_RMSE=f"{local['section_in_domain']['frequency_rmse_hz']:.7g}",
        SECTION_MAX=f"{local['section_in_domain']['frequency_max_abs_hz']:.7g}",
        DERIVATIVES=deriv,
        SEED=str(local["selected_seed"]),
        EPOCH=str(local["selected_epoch"]),
        DEGREES=str(local["formulas"]["degree"]),
        COEFFICIENTS=coefficients,
        EXTRACTION=f"{local['formulas']['max_extraction_error_dimensionless']:.3g}",
        ROUNDED_POINT=fmt(local["paper_reference_frequencies_hz"]),
        LOCAL_TABLE=metric_table(local),
        BROAD_TABLE=metric_table(broad),
        BROAD_POINT=fmt(broad["reference_point"]["asn_frequencies_hz"]),
        SEEDS=seeds,
        BROAD_VIOLATIONS=str(broad["evaluations"]["ASN"]["ordering_violations"]),
        BOUNDARY_ERRORS=fmt(
            [r["boundary_challenge"]["frequency_max_abs_hz"] for r in (local, broad)]
        ),
        BOUNDARY_VIOLATIONS=", ".join(
            str(r["boundary_challenge"]["ordering_violations"]) for r in (local, broad)
        ),
    )
    for k, v in repl.items():
        text = text.replace("@@" + k + "@@", v)
    if "@@" in text:
        raise ValueError("Unfilled placeholder")
    return text


def manuscript(physics, results):
    r = results["local"]
    ev = r["evaluations_extended"]
    point = r["reference_point"]
    fmt = lambda a: ", ".join(f"{v:.6f}" for v in a)
    source = r"""% Generated from completed homotaurine training and independent analytical reference.
\subsection{A second molecular application: homotaurine}
The same normalization and four-state secular model were applied to the
supplied illustrative homotaurine vector
$(J_A,J_M,J_X,d,e)=(-12.743923,-13.969277,-14.109177,3.093406,2.344390)$ Hz.
The temperature label of this historical fit is unresolved (289/298 K);
these calculations are not presented as a new experimental refit.
The normalized inputs are $(x_A,x_X,u,v)=((J_A-J_M)/s,(J_X-J_M)/s,d/s,e/s)$,
where $s=-J_M>0$ and $E_i=(-3J_A+J_M-3J_X)/4+s\lambda_i$.
Three unequal geminals and type-II mixing are retained.

As in the DSS experiment, a (4,16,4) ASN with $\{0,x,x^2\}$ activations
was trained on 100000 numerically diagonalized Hamiltonians, with disjoint
80000/10000/10000 training/validation/test splits and five initialization seeds.
Each geminal varied independently by $\pm0.3$ Hz and each coupling difference
by $\pm20\%$ about homotaurine. Training used Adam ($10^{-3}$), MAE,
batch size 4096, 900 epochs of softmax with $0.1L_{0.5}$ and 100 epochs of
sparsemax without the penalty. Input/output standardization used only training
statistics and was undone algebraically in the extracted expressions.
Validation eigenvalue RMSE alone selected the checkpoint and representative seed.
Eigenvalue order was preserved in the target labels; predictions were not sorted afterward.

@@RESULTS@@

With geminals fixed and $d=3.093406t$, $e=2.344390t$, the analytical
frequencies follow from the ordered roots of $z^3-C_1z^2+C_2z-C_3=0$, where
\[C_1=2.852388810881+0.077202373264t^2,\]
\[C_2=2.701400997954+0.073465038627t^2+0.001381140469t^4,\quad C_3=0.849012187073.\]
Writing $\omega_i=s\sqrt{z_i}$ gives $f_{12}=\omega_3-\omega_2$,
$f_{23}=\omega_2-\omega_1$, and $f_{13}=f_{12}+f_{23}$.
Substitution into the trained four-input ASN, without an additional section fit, gives
@@FORMULAS@@
These are finite-domain learned approximations for $0.8\le t\le1.2$,
not exact perturbative Taylor coefficients. Spectral overlays use common
illustrative local-singlet amplitudes and 0.1 Hz Lorentzian FWHM, changing only
the line centers between exact and learned predictions.

Dropping type-II mixing shifts a principal line by 0.053710 Hz; replacing
geminals by their mean shifts a line by 0.278275 Hz. The second-order effective
three-state Hamiltonian has a maximum line error of 0.000287194 Hz at this point.
The fitted end-to-end difference $q=0.027306$ Hz changes a line by up to
0.018095 Hz and produces matched-parity separations up to 0.036090 Hz.
It should be included in precision comparisons with measured spectra; it is
omitted explicitly in both the analytical and learned four-input benchmark.
Full parity-resolved calculations and approximation errors accompany the comparison.

These results demonstrate explicit learned molecular approximations in a
specified domain. Polynomial and effective-Hamiltonian baselines are retained;
no intrinsic accuracy advantage over an appropriate fixed basis is claimed.
The broad-box run, rounded-coefficient errors, seed variation and decoupled-boundary
ordering checks are reported in the supplementary material.
"""
    result = (
        f"The selected local ASN (seed {r['selected_seed']}) achieved frequency RMSE "
        f"{ev['ASN']['frequency_rmse_hz']:.7g} Hz and maximum line error {ev['ASN']['frequency_max_abs_hz']:.7g} Hz "
        f"on 10000 held-out samples, with {ev['ASN']['ordering_violations']} ordering violations. "
        f"Quadratic and quartic least-squares RMSEs on the same data were {ev['OLS degree 2']['frequency_rmse_hz']:.7g} and "
        f"{ev['OLS degree 4']['frequency_rmse_hz']:.7g} Hz. "
        "At the supplied homotaurine reference, the analytical frequencies were "
        + fmt(point["exact_frequencies_hz"])
        + " Hz and the ASN frequencies were "
        + fmt(point["asn_frequencies_hz"])
        + f" Hz. Three-decimal coefficient rounding gave a local frequency RMSE of {ev['ASN paper-style 3-decimal rounding']['frequency_rmse_hz']:.7g} Hz."
    )
    formulas = polynomial_lines(r["formulas"]["cross_section_coefficients_hz"])
    formulas = "\n".join(r"\[" + part + r"\]" for part in formulas.split("$$")[1::2])
    source = source.replace("@@RESULTS@@", result).replace("@@FORMULAS@@", formulas)
    (OUT / "homotaurine_manuscript_addition.tex").write_text(source)


def main():
    physics = analytical_artifacts()
    results = {domain: analyze(domain, OUT, P) for domain in ("local", "broad")}
    figures(results, OUT, P, "Homotaurine")
    write_report(
        build_text(physics, results),
        OUT,
        "homotaurine_asn_comparison",
        "Homotaurine: analytical and trained ASN results",
    )
    manuscript(physics, results)
    for domain, r in results.items():
        print(
            domain,
            "seed",
            r["selected_seed"],
            "ASN",
            r["evaluations"]["ASN"],
            "point",
            r["reference_point"],
            flush=True,
        )
    print("Homotaurine analytical and trained-model reports written.")


if __name__ == "__main__":
    main()
