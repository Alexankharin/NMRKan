"""Exact normalized DSS solution and reproducible spectral sections.

Numerics require NumPy; --figures additionally requires Matplotlib.
Run: python -m nmrkan.physics.real_molecules --figures
The supplied asn_three_ch2.py remains the independent 4x4 reference.
All energies, frequencies, and couplings are in Hz (not rad/s).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nmrkan.physics.three_ch2 import (
    BASIS_EVEN,
    BASIS_ODD,
    EXAMPLES,
    hamiltonian,
    normalize_parameters,
)


def _parameters(parameters):
    p = np.asarray(parameters, dtype=float)
    if p.shape != (5,) or not np.isfinite(p).all() or np.any(p[:3] >= 0):
        raise ValueError("Need five finite parameters and three negative geminal J's.")
    return p


def mode_matrix(parameters):
    """Dimensionless B: its singular values are the three normal-mode energies/s."""
    a, b, c, d, e = _parameters(parameters)
    return (np.diag([a, b, c]) + np.diag([d, e], 1)) / (-b)


def cubic_invariants(parameters):
    """Coefficients C1,C2,C3 of z^3-C1*z^2+C2*z-C3 for squared modes."""
    a, b, c, d, e = _parameters(parameters)
    a, b, c, d, e = np.array([a, b, c, d, e]) / (-b)
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
    return np.array([c1, c2, c3])


def symmetric_cubic_roots(matrix):
    """Ordered real roots for a symmetric 3x3 matrix, by the trigonometric formula.

    Evaluate centered invariants directly to avoid cancellation in C1^2-3*C2.
    Diagonal matrices are handled directly, including exact double/triple roots.
    Very near a repeated root the arccos representation is ill-conditioned;
    use the symmetric eigensolver there, rather than manufacture a splitting.
    This fallback concerns numerical evaluation, not the analytical identity.
    """
    M = np.asarray(matrix, dtype=float)
    if M.shape != (3, 3) or not np.isfinite(M).all() or not np.allclose(M, M.T):
        raise ValueError("Expected a finite symmetric 3x3 matrix.")
    off = M - np.diag(np.diag(M))
    if np.count_nonzero(off) == 0:
        return np.sort(np.diag(M))
    mean = np.trace(M) / 3.0
    centered = M - mean * np.eye(3)
    radius = np.sqrt(np.sum(centered * centered) / 6.0)
    if radius == 0:
        return np.full(3, mean)
    arg = np.linalg.det(centered / radius) / 2.0
    if 1 - abs(arg) < 1e-10:
        return np.linalg.eigvalsh(M)
    theta = np.arccos(np.clip(arg, -1.0, 1.0)) / 3.0
    return mean + 2 * radius * np.cos(theta + np.array([2, -2, 0]) * np.pi / 3)


def normal_modes(parameters, method="analytic"):
    B = mode_matrix(parameters)
    if method == "analytic":
        roots = symmetric_cubic_roots(B @ B.T)
        if roots[0] < -1e-12 * max(1.0, roots[-1]):
            raise ArithmeticError("Squared mode unexpectedly negative.")
        modes = np.sqrt(np.maximum(roots, 0.0))
    elif method == "svd":
        modes = np.sort(np.linalg.svd(B, compute_uv=False))
    else:
        raise ValueError("method must be 'analytic' or 'svd'.")
    return -float(parameters[1]) * modes


def analytical_eigenvalues(parameters, method="analytic"):
    """E0<=E1<=E2<=E3 in the EVEN block, exact for q=0 and negative geminals.

    These ordered surfaces match the paper's physical-energy convention.
    A fixed order is not a claim of globally smooth eigenvector identity.
    """
    p = _parameters(parameters)
    u, v, w = normal_modes(p, method)
    center = -sum(p[:3]) / 4
    return center + np.array([-u - v - w, u + v - w, u - v + w, -u + v + w]) / 2


def principal_gaps(energies):
    """Identity labels f12,f23,f13 among three ordered principal levels.

    Do not sort the first two frequencies by their magnitudes: they can cross.
    """
    w = np.asarray(energies)
    if w.shape != (3,) or np.any(np.diff(w) < -1e-12):
        raise ValueError("Expected three ordered principal energies.")
    return np.array([w[1] - w[0], w[2] - w[1], w[2] - w[0]])


def analytical_frequencies(parameters):
    u, v, w = normal_modes(parameters)
    return np.array([w - v, v - u, w - u])


def cross_section_taylor(parameters, at=1.0):
    """Analytic value/first derivative/half second derivative along d,e -> t*d,t*e.

    Uses Hermitian eigenvalue derivative identities, not a polynomial fit or
    finite differences. Rows correspond to f12,f23,f13; columns to 1,dt,dt^2.
    Nondegenerate normal modes are required at the expansion point.
    """
    p = _parameters(parameters)
    scale = -p[1]
    diagonal = np.diag(p[:3] / scale)
    V = np.diag(p[3:] / scale, 1)
    B = diagonal + at * V
    M = B @ B.T
    M1 = V @ B.T + B @ V.T
    M2 = 2 * V @ V.T
    z, U = np.linalg.eigh(M)
    if np.min(np.diff(z)) < 1e-10:
        raise ValueError("Taylor formula requires distinct normal modes.")
    Q1, Q2 = U.T @ M1 @ U, U.T @ M2 @ U
    z1 = np.diag(Q1)
    z2 = np.diag(Q2).copy()
    for k in range(3):
        z2[k] += 2 * sum(Q1[l, k] ** 2 / (z[k] - z[l]) for l in range(3) if l != k)
    modes = np.column_stack(
        (
            scale * np.sqrt(z),
            scale * z1 / (2 * np.sqrt(z)),
            scale * (z2 / (2 * np.sqrt(z)) - z1 * z1 / (4 * z**1.5)) / 2,
        )
    )
    return np.array([modes[2] - modes[1], modes[1] - modes[0], modes[2] - modes[0]])


def effective_three_state(parameters):
    """Hermitian second-order elimination; type-I block diagonalized exactly.

    Includes induced off-diagonal terms, including at a degeneracy inside A.
    Expansion is in coupling to the outlier, not in the internal level gaps.
    """
    H = hamiltonian(*_parameters(parameters))
    A, v, b = H[:3, :3], H[:3, 3], H[3, 3]
    levels, U = np.linalg.eigh(A)
    if np.min(np.abs(levels - b)) < 1e-10:
        raise ValueError("Outlier is not separated; elimination is invalid.")
    v = U.T @ v
    reciprocal = 1 / (levels - b)
    return (
        np.diag(levels)
        + np.outer(v, v) * (reciprocal[:, None] + reciprocal[None, :]) / 2
    )


def singlet_observable(site="A", parity="even"):
    """Traceless local singlet population within one parity sector.

    This is an illustrative preparation/readout model, not a poly-SLIC simulation.
    """
    if site not in ("A", "M", "X") or parity not in ("even", "odd"):
        raise ValueError("site A/M/X and parity even/odd required.")
    basis = BASIS_EVEN if parity == "even" else BASIS_ODD
    index = ("A", "M", "X").index(site)
    return np.diag([float(state[index] == "S") - 0.5 for state in basis])


def spectral_lines(H, rho, observable):
    """All six pair contributions to Tr[O exp(-i2piHt) rho exp(i2piHt)].

    Real signal = DC + sum(cosine*cos(2pi*f*t) + sine*sin(2pi*f*t)).
    At a degeneracy only the sum over equal-frequency rows is invariant under
    rotations of degenerate eigenvectors. Zero-frequency rows contribute to DC.
    """
    H, rho, observable = map(np.asarray, (H, rho, observable))
    for matrix in (H, rho, observable):
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Expected finite 4x4 matrices.")
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("H, rho and observable must be Hermitian.")
    E, U = np.linalg.eigh(H)
    r = U.conj().T @ rho @ U
    o = U.conj().T @ observable @ U
    lines = []
    for i in range(4):
        for j in range(i + 1, 4):
            z = r[i, j] * o[j, i]
            lines.append(
                dict(
                    i=i,
                    j=j,
                    frequency_hz=float(E[j] - E[i]),
                    cosine=float(2 * z.real),
                    sine=float(-2 * z.imag),
                )
            )
    return float(np.real(np.sum(np.diag(r) * np.diag(o)))), lines


def lorentzian_spectrum(frequencies, lines, fwhm=0.1):
    """Positive-frequency absorption profile with line areas = cosine amplitudes.

    Pure cosine lines only; mirror lines/DC and dispersive sine terms are omitted.
    fwhm=0.1 Hz is a literature-scale illustration, not a measured DSS fit.
    """
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("fwhm must be positive.")
    if any(abs(line["sine"]) > 1e-12 for line in lines):
        raise ValueError("Absorption profile requires real cosine amplitudes.")
    f = np.asarray(frequencies)
    gamma = fwhm / 2
    return sum(
        line["cosine"] * gamma / np.pi / ((f - line["frequency_hz"]) ** 2 + gamma**2)
        for line in lines
        if line["frequency_hz"] > 1e-12
    )


def _write_rows(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_results(outdir, figures=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results, comparison_rows, line_rows = {}, [], []
    for fullname, example in EXAMPLES.items():
        name = fullname.split(" (")[0]
        p = np.array(example["parameters"])
        H = hamiltonian(*p)
        E = analytical_eigenvalues(p)
        f = analytical_frequencies(p)
        features, scale, reference = normalize_parameters(*p)
        equal = p.copy()
        equal[:3] = np.mean(p[:3])
        variants = {
            "exact nearest-neighbor": f,
            "drop type-II (bare 3x3)": principal_gaps(np.linalg.eigvalsh(H[:3, :3])),
            "second-order effective 3x3": principal_gaps(
                np.linalg.eigvalsh(effective_three_state(p))
            ),
            "replace geminals by their mean": analytical_frequencies(equal),
            "include q, even parity": principal_gaps(
                np.linalg.eigvalsh(hamiltonian(*p, dJAX=example["dJAX"]))[1:]
            ),
            # Reflection reverses internal order in the odd block. Reverse the
            # two adjacent gaps here to retain the EVEN-equivalent line labels.
            "include q, odd parity (matched)": principal_gaps(
                np.linalg.eigvalsh(hamiltonian(*p, dJAX=example["dJAX"], parity="odd"))[
                    :3
                ]
            )[[1, 0, 2]],
        }
        for variant, values in variants.items():
            comparison_rows.append(
                dict(
                    molecule=name,
                    model=variant,
                    f12_hz=values[0],
                    f23_hz=values[1],
                    f13_hz=values[2],
                    max_abs_error_vs_q0_hz=max(abs(values - f)),
                )
            )
        results[name] = dict(
            parameters_hz=p.tolist(),
            q_hz=example["dJAX"],
            provenance=fullname,
            features=features.tolist(),
            scale_hz=scale,
            reference_hz=reference,
            modes_hz=normal_modes(p).tolist(),
            eigenvalues_hz=E.tolist(),
            normalized_eigenvalues=((E - reference) / scale).tolist(),
            frequencies_hz=f.tolist(),
            variants={k: v.tolist() for k, v in variants.items()},
        )
        taylor = cross_section_taylor(p)
        errors = []
        for t in np.linspace(0.9, 1.1, 401):
            pt = p.copy()
            pt[3:] *= t
            errors.append(
                np.abs(
                    taylor @ np.array([1, t - 1, (t - 1) ** 2])
                    - analytical_frequencies(pt)
                )
            )
        results[name]["taylor_about_t1_hz"] = taylor.tolist()
        results[name]["taylor_max_error_0p9_to_1p1_hz"] = np.max(
            errors, axis=0
        ).tolist()
        for q in (0.0, example["dJAX"]):
            for parity in ("even", "odd"):
                for site in ("A", "M", "X"):
                    O = singlet_observable(site, parity)
                    _, lines = spectral_lines(
                        hamiltonian(*p, dJAX=q, parity=parity), O, O
                    )
                    for line in lines:
                        line_rows.append(
                            dict(
                                molecule=name, q_hz=q, parity=parity, site=site, **line
                            )
                        )
        rows = []
        for t in np.linspace(0, 1.5, 601):
            pt = p.copy()
            pt[3:] *= t
            Et = analytical_eigenvalues(pt)
            ft = analytical_frequencies(pt)
            reduced = principal_gaps(np.linalg.eigvalsh(hamiltonian(*pt)[:3, :3]))
            effective = principal_gaps(np.linalg.eigvalsh(effective_three_state(pt)))
            row = dict(t=t, yAM=t * features[2], yMX=t * features[3])
            row.update({f"E{i}_hz": v for i, v in enumerate(Et)})
            row.update(
                {f"lambda{i}": v for i, v in enumerate((Et - reference) / scale)}
            )
            for prefix, values in (
                ("f", ft),
                ("bare_f", reduced),
                ("effective_f", effective),
            ):
                row.update(
                    {f"{prefix}{label}_hz": v for label, v in zip((12, 23, 13), values)}
                )
            rows.append(row)
        _write_rows(outdir / f"{name.lower()}_cross_section.csv", rows)
        if name == "DSS":
            axis = np.linspace(0, 36, 18001)
            O = singlet_observable("A")
            _, lines = spectral_lines(H, O, O)
            spectrum = lorentzian_spectrum(axis, lines)
            _write_rows(
                outdir / "dss_illustrative_spectrum.csv",
                [
                    dict(frequency_hz=freq, absorption=value, fwhm_hz=0.1)
                    for freq, value in zip(axis, spectrum)
                ],
            )
        if figures and name == "DSS":
            make_figures(outdir, p, rows, results[name])
    (outdir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    _write_rows(outdir / "model_comparison.csv", comparison_rows)
    _write_rows(outdir / "illustrative_transition_amplitudes.csv", line_rows)
    return results


def make_figures(outdir, p, rows, result, name="DSS"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = ["#0077aa", "#d1495b", "#308050", "#7851a9"]

    def save(fig, name):
        for ext in ("pdf", "svg", "png"):
            fig.savefig(outdir / f"{name}.{ext}", dpi=170)
        plt.close(fig)

    slug = name.lower()

    t = np.array([r["t"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i in range(4):
        ax[0, 0].plot(
            t, [r[f"lambda{i}"] for r in rows], color=colors[i], label=rf"$\lambda_{i}$"
        )
    ax[0, 0].set(
        xlabel=r"Coupling scale $t$",
        ylabel="Normalized energy",
        title="A  Four ordered eigenvalue branches",
    )
    ax[0, 0].legend(ncols=2)
    for color, label in zip(colors, (12, 23, 13)):
        exact = np.array([r[f"f{label}_hz"] for r in rows])
        ax[0, 1].plot(t, exact, color=color, label=rf"$f_{{{label}}}$")
        ax[1, 0].plot(
            t, np.array([r[f"bare_f{label}_hz"] for r in rows]) - exact, color=color
        )
        ax[1, 0].plot(
            t,
            np.array([r[f"effective_f{label}_hz"] for r in rows]) - exact,
            color=color,
            ls="--",
        )
    ax[0, 1].set(
        xlabel=r"Coupling scale $t$",
        ylabel="Frequency / Hz",
        title="B  Principal transitions",
    )
    ax[0, 1].legend()
    ax[1, 0].axhline(0, color="0.7", lw=0.7)
    ax[1, 0].set(
        xlabel=r"Coupling scale $t$",
        ylabel="Approximation minus exact / Hz",
        title="C  Solid: bare 3×3; dashed: effective 3×3",
    )
    f = np.linspace(0, max(result["frequencies_hz"]) + 1, 10001)
    O = singlet_observable("A")
    _, lines = spectral_lines(hamiltonian(*p), O, O)
    y = lorentzian_spectrum(f, lines)
    ax[1, 1].plot(f, y / y.max(), color="#1d3557")
    for label, frequency, color in zip((12, 23, 13), result["frequencies_hz"], colors):
        ax[1, 1].axvline(frequency, color=color, ls=":", lw=0.8)
        ax[1, 1].text(
            frequency + 0.08,
            0.80 if label != 23 else 0.95,
            f"{frequency:.4f}",
            color=color,
            rotation=90,
            va="top",
        )
    ax[1, 1].set(
        xlabel="ZQ frequency / Hz",
        ylabel="Normalized absorption",
        title=f"D  {name} at t = 1; illustrative amplitudes",
    )
    for axis in (ax[0, 0], ax[0, 1], ax[1, 0]):
        axis.axvline(1, color="0.4", ls=":")
    fig.suptitle(
        f"{name}: exact normalized model and three-line spectral section\nFixed geminals; d(t)={p[3]:.6f}t Hz, e(t)={p[4]:.6f}t Hz; q=0",
        fontsize=13,
    )
    save(fig, f"{slug}_cross_section")

    features, scale, reference = normalize_parameters(*p)
    grid = np.linspace(0, 7, 65)
    d, e = np.meshgrid(grid, grid)
    surfaces = np.empty(d.shape + (4,))
    surface_rows = []
    for i in range(len(grid)):
        for j in range(len(grid)):
            pt = p.copy()
            pt[3:] = d[i, j], e[i, j]
            E = analytical_eigenvalues(pt)
            surfaces[i, j] = (E - reference) / scale
            surface_rows.append(
                dict(
                    d_hz=d[i, j],
                    e_hz=e[i, j],
                    yAM=d[i, j] / scale,
                    yMX=e[i, j] / scale,
                    **{f"lambda{k}": v for k, v in enumerate(surfaces[i, j])},
                )
            )
    _write_rows(outdir / f"{slug}_eigenvalue_surface.csv", surface_rows)
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    for k in range(4):
        axis = fig.add_subplot(2, 2, k + 1, projection="3d", computed_zorder=False)
        axis.plot_surface(
            d / scale,
            e / scale,
            surfaces[:, :, k],
            cmap="viridis",
            alpha=0.88,
            rasterized=True,
            zorder=1,
        )
        valid = t <= min(7 / p[3], 7 / p[4])
        axis.plot(
            t[valid] * features[2],
            t[valid] * features[3],
            np.array([r[f"lambda{k}"] for r in rows])[valid],
            color="#d1495b",
            lw=2,
            zorder=3,
        )
        axis.scatter(
            [features[2]],
            [features[3]],
            [result["normalized_eigenvalues"][k]],
            color="black",
            s=28,
            zorder=4,
        )
        axis.set(
            xlabel=r"$d/s$",
            ylabel=r"$e/s$",
            zlabel=rf"$\lambda_{k}$",
            title=f"Branch {k}",
        )
        axis.view_init(24, -125)
    fig.suptitle(
        f"{name} geminals fixed: two-dimensional eigenvalue surfaces\nRed: chosen cross-section; black: supplied {name} parameter point",
        fontsize=13,
    )
    save(fig, f"{slug}_eigenvalue_surfaces")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for line in lines:
        principal = line["i"] > 0
        axes[0].vlines(
            line["frequency_hz"],
            0,
            line["cosine"],
            color="#0077aa" if principal else "#d1495b",
            lw=2,
        )
    axes[0].set(
        xlabel="Frequency / Hz",
        ylabel="Cosine amplitude",
        title="A  All six transitions: local A singlet autocorrelation",
        xlim=(0, 36),
    )
    f = np.linspace(
        min(result["frequencies_hz"][:2]) - 0.25,
        max(result["frequencies_hz"][:2]) + 0.25,
        6001,
    )
    for q, style, label in (
        (0, "-", "q = 0"),
        (result["q_hz"], "--", "Fitted q; equally weighted parity blocks"),
    ):
        y = np.zeros_like(f)
        for parity in ("even", "odd"):
            O = singlet_observable("A", parity)
            _, part = spectral_lines(hamiltonian(*p, dJAX=q, parity=parity), O, O)
            y += 0.5 * lorentzian_spectrum(f, part)
        axes[1].plot(f, y, label=label, ls=style)
    axes[1].set(
        xlabel="Frequency / Hz",
        ylabel="Absorption / arbitrary units",
        title="B  q correction at illustrative 0.1 Hz FWHM",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Illustrative readout: amplitudes are not an experimental poly-SLIC fit",
        fontsize=12,
    )
    save(fig, f"{slug}_observability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/molecules/analytical")
    )
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    results = build_results(args.output, args.figures)
    for name, result in results.items():
        print(name, "f12, f23, f13 (Hz):", result["frequencies_hz"])
    print("Written to", args.output)


if __name__ == "__main__":
    main()
