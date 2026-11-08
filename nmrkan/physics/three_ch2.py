"""Three-CH2 spin-chain ZQ Hamiltonians for ASN (couplings and energies in Hz).

Model: chemically equivalent protons within each pair; full intrapair scalar
couplings, secular (Iz Iz) interpair couplings; no RF during evolution.
S=(alpha beta-beta alpha)/sqrt(2), T=T0=(alpha beta+beta alpha)/sqrt(2).

Delta J is the FULL difference J_AM-J_AM', NOT its half. The matrix element
is Delta J/2. hamiltonian() includes type-I AND type-II mixing. It is exact
within the stated secular model, not an exact full molecular Hamiltonian.

Run: python asn_three_ch2.py --check
Optional training data: python asn_three_ch2.py --samples 10000 --csv data.csv
Only dependency: numpy. No training or pickle loading occurs in this module.
"""

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


BASIS_EVEN = ("TSS", "STS", "SST", "TTT")  # even number of singlets
BASIS_ODD = ("STT", "TST", "TTS", "SSS")  # odd number of singlets
PARAMETER_NAMES = ("JAA", "JMM", "JXX", "dJAM", "dJMX")
FEATURE_NAMES = ("xA", "xX", "yAM", "yMX")
GEMINAL_BOUNDS_HZ = (-15.0, -12.5)
DELTA_BOUNDS_HZ = (0.1, 7.0)

# Historical fitted PHYSICAL rows, not newly measured or independently refitted.
# A, M, X labels follow those files. Full differences calculated from those rows.
EXAMPLES = {
    "DSS (31 July 2026 email; fit labeled 298 K)": {
        "parameters": (-13.758727, -13.408159, -14.121958, 5.269170, 6.597217),
        "dJAX": 0.002490,
    },
    "Homotaurine (15 August 2026 email; temperature label unresolved)": {
        "parameters": (-12.743923, -13.969277, -14.109177, 3.093406, 2.344390),
        "dJAX": 0.027306,
    },
}


def hamiltonian(JAA, JMM, JXX, dJAM, dJMX, dJAX=0.0, *, parity="even"):
    """Return one real symmetric 4x4 parity block, in Hz.

    even basis: TSS, STS, SST, TTT; odd basis: STT, TST, TTS, SSS.
    JAA=J_AA', JMM=J_MM', JXX=J_XX'.
    dJAM=J_AM-J_AM', assuming J_AM=J_A'M', J_AM'=J_A'M;
    likewise for dJMX and optional non-nearest-neighbor dJAX.
    Set dJAX=0 for the requested five-parameter nearest-neighbor model.
    Absolute eigenvalue signs depend on the chosen energy zero: keep all four.
    """
    values = np.asarray((JAA, JMM, JXX, dJAM, dJMX, dJAX), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All coupling parameters must be finite.")
    a, b, c, d, e, f = values
    total = a + b + c
    if parity == "even":
        diagonal = -0.75 * total + np.array((a, b, c, total))
    elif parity == "odd":
        diagonal = 0.25 * total - np.array((a, b, c, total))
    else:
        raise ValueError("parity must be 'even' or 'odd'.")
    H = np.diag(diagonal)
    for i, j, coupling in (
        (0, 1, d),
        (1, 2, e),
        (0, 3, e),
        (2, 3, d),
        (0, 2, f),
        (1, 3, f),
    ):
        H[i, j] = H[j, i] = coupling / 2.0
    return H


def principal_frequencies(H):
    """Three gaps among the upper three levels, sorted, in input energy units.

    For the stated negative-geminal training domain these levels evolve from
    TSS, STS, SST in the even block.
    These are candidate line positions, not predictions of line intensities.
    For arbitrary parameters use eigenvectors/observables to identify lines.
    The odd block's principal levels are its LOWER three, so pass an even block
    here; use principal_frequencies_odd() for the odd block.
    """
    w = np.linalg.eigvalsh(H)[1:]
    return np.sort((w[1] - w[0], w[2] - w[1], w[2] - w[0]))


def principal_frequencies_odd(H):
    """Three gaps among the lower three levels of the odd block."""
    w = np.linalg.eigvalsh(H)[:3]
    return np.sort((w[1] - w[0], w[2] - w[1], w[2] - w[0]))


def normalize_parameters(JAA, JMM, JXX, dJAM, dJMX):
    """Return four dimensionless inputs, positive scale, and energy reference.

    s=-JMM>0; E_ref=E_STS=(-3*JAA+JMM-3*JXX)/4.
    H_even = E_ref*I + s*K. Eigenvalues in Hz are E_ref+s*lambda(K).
    Frequency differences in Hz are s*abs(lambda_i-lambda_j).
    """
    parameters = np.asarray((JAA, JMM, JXX, dJAM, dJMX), dtype=float)
    if not np.isfinite(parameters).all() or JMM >= 0:
        raise ValueError("Need finite parameters and JMM < 0.")
    s = -float(JMM)
    x = np.array(((JAA - JMM) / s, (JXX - JMM) / s, dJAM / s, dJMX / s))
    reference = (-3.0 * JAA + JMM - 3.0 * JXX) / 4.0
    return x, s, reference


def dimensionless_matrix(xA, xX, yAM, yMX):
    """Four-input five-physical-parameter matrix; dJAX is zero."""
    return np.array(
        [
            [xA, yAM / 2.0, 0.0, yMX / 2.0],
            [yAM / 2.0, 0.0, yMX / 2.0, 0.0],
            [0.0, yMX / 2.0, xX, yAM / 2.0],
            [yMX / 2.0, 0.0, yAM / 2.0, -2.0 + xA + xX],
        ],
        dtype=float,
    )


def make_dataset(
    n, seed=0, geminal_bounds=GEMINAL_BOUNDS_HZ, delta_bounds=DELTA_BOUNDS_HZ
):
    """Sample physical couplings, then normalize; no separate feature boxes.

    targets are ALL four sorted dimensionless eigenvalues. If training just
    the dominant one-T manifold, explicitly select targets[:, 1:].
    Sorting fixes energy order, not adiabatic state identity near crossings.
    Independent samples from these boxes are a training design, not a joint
    physical probability distribution or fitted confidence intervals.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer.")
    g0, g1 = geminal_bounds
    d0, d1 = delta_bounds
    if not (g0 < g1 < 0 and 0 <= d0 < d1):
        raise ValueError("Need ordered negative geminal and nonnegative delta bounds.")
    rng = np.random.default_rng(seed)
    parameters = np.column_stack(
        (rng.uniform(g0, g1, (n, 3)), rng.uniform(d0, d1, (n, 2)))
    )
    features = np.empty((n, 4))
    targets = np.empty((n, 4))
    frequencies = np.empty((n, 3))
    scales = np.empty(n)
    references = np.empty(n)
    for i, p in enumerate(parameters):
        features[i], scales[i], references[i] = normalize_parameters(*p)
        targets[i] = np.linalg.eigvalsh(dimensionless_matrix(*features[i]))
        w = targets[i, 1:]
        frequencies[i] = scales[i] * np.sort((w[1] - w[0], w[2] - w[1], w[2] - w[0]))
    return dict(
        parameters_hz=parameters,
        features=features,
        targets=targets,
        frequencies_hz=frequencies,
        scale_hz=scales,
        reference_hz=references,
    )


def _pair_product_signs(basis):
    """Integer Q with normalized six-spin basis columns Q/sqrt(8)."""
    Q = np.zeros((64, 4), dtype=np.int64)
    for col, labels in enumerate(basis):
        for pair_bits in product((0, 1), repeat=3):
            bits = tuple(b for p in pair_bits for b in (p, 1 - p))
            row = sum(b << (5 - i) for i, b in enumerate(bits))
            Q[row, col] = (-1) ** sum(
                p for p, label in zip(pair_bits, labels) if label == "S"
            )
    return Q


def _physical_coefficient_times_four(parameter):
    """Independent 64x64 six-proton spin operator coefficient, scaled by 4.

    Integer arithmetic: 4*(Ia.Ia') for a geminal coefficient, or
    4*(Iz_a-Iz_a')*(Iz_b-Iz_b')/2 for a full Delta-J coefficient.
    """
    M = np.zeros((64, 64), dtype=np.int64)
    for row in range(64):
        bits = [(row >> (5 - i)) & 1 for i in range(6)]
        z = [1 - 2 * b for b in bits]  # twice the physical Iz eigenvalue
        if parameter < 3:
            a, b = 2 * parameter, 2 * parameter + 1
            M[row, row] = z[a] * z[b]
            if bits[a] != bits[b]:
                flipped = row ^ (1 << (5 - a)) ^ (1 << (5 - b))
                M[flipped, row] = 2
        else:
            a, b = ((0, 1), (1, 2), (0, 2))[parameter - 3]
            M[row, row] = ((z[2 * a] - z[2 * a + 1]) * (z[2 * b] - z[2 * b + 1])) // 2
    return M


def self_check():
    """Exact coefficient projection plus numerical spectrum/scale checks."""
    coefficient_checks = 0
    for parity, basis in (("even", BASIS_EVEN), ("odd", BASIS_ODD)):
        Q = _pair_product_signs(basis)
        np.testing.assert_array_equal(Q.T @ Q, 8 * np.eye(4, dtype=np.int64))
        for k in range(6):
            unit = np.eye(6)[k]
            expected = hamiltonian(*unit, parity=parity)
            M4 = _physical_coefficient_times_four(k)
            # Qnorm=Q/sqrt(8), physical coefficient=M4/4: divisor is 32.
            np.testing.assert_array_equal(
                Q.T @ M4 @ Q, np.rint(32 * expected).astype(np.int64)
            )
            # Also prove the subspace invariant, not merely a projection.
            np.testing.assert_array_equal(
                M4 @ Q, Q @ np.rint(4 * expected).astype(np.int64)
            )
            coefficient_checks += 1
    np.testing.assert_array_equal(
        _pair_product_signs(BASIS_EVEN).T @ _pair_product_signs(BASIS_ODD),
        np.zeros((4, 4)),
    )
    rng = np.random.default_rng(20261018)
    D = np.diag((1.0, -1.0, 1.0, -1.0))
    for _ in range(100):
        p = np.r_[rng.uniform(-15, -12.5, 3), rng.uniform(0.1, 7, 2)]
        H = hamiltonian(*p)
        Ho = hamiltonian(*p, parity="odd")
        np.testing.assert_allclose(
            Ho, -D @ H @ D - 0.5 * sum(p[:3]) * np.eye(4), atol=2e-14
        )
        np.testing.assert_allclose(
            principal_frequencies(H), principal_frequencies_odd(Ho), atol=5e-14
        )
        x, scale, reference = normalize_parameters(*p)
        K = dimensionless_matrix(*x)
        np.testing.assert_allclose(H, reference * np.eye(4) + scale * K, atol=2e-14)
        np.testing.assert_allclose(hamiltonian(*(3.7 * p)), 3.7 * H, atol=3e-14)
    p = (-13.0, -14.0, -15.0, 0.0, 0.0)
    np.testing.assert_array_equal(hamiltonian(*p), np.diag((18.5, 17.5, 16.5, -10.5)))
    np.testing.assert_allclose(
        principal_frequencies(hamiltonian(-14, -14, -14, 6.06, 6.06)),
        (3.92071840, 4.57615388, 8.49687228),
        atol=1e-8,
        rtol=0,
    )
    ds = make_dataset(8, seed=71)
    for p, x, w, scale, reference in zip(
        ds["parameters_hz"],
        ds["features"],
        ds["targets"],
        ds["scale_hz"],
        ds["reference_hz"],
    ):
        np.testing.assert_allclose(
            np.linalg.eigvalsh(hamiltonian(*p)), reference + scale * w, atol=3e-14
        )
    return (
        f"PASS: {coefficient_checks} exact 64-to-4 spin-operator coefficient "
        "and invariance checks; 100 parity/normalization/scaling cases; "
        "zero-coupling and equal-coupling limits; dataset reconstruction."
    )


def _write_csv(path, data):
    headers = (
        list(PARAMETER_NAMES)
        + list(FEATURE_NAMES)
        + [f"lambda_{i}" for i in range(4)]
        + ["f_small_hz", "f_middle_hz", "f_sum_hz", "scale_hz", "reference_hz"]
    )
    columns = np.column_stack(
        (
            data["parameters_hz"],
            data["features"],
            data["targets"],
            data["frequencies_hz"],
            data["scale_hz"],
            data["reference_hz"],
        )
    )
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run independent checks.")
    parser.add_argument(
        "--samples", type=int, default=0, help="Generate this many samples."
    )
    parser.add_argument(
        "--csv", type=Path, help="Explicit output path for generated samples."
    )
    args = parser.parse_args()
    if args.check:
        print(self_check())
    np.set_printoptions(precision=8, suppress=True, linewidth=110)
    print("Even basis:", BASIS_EVEN, "; all matrices and eigenvalues in Hz.")
    for name, example in EXAMPLES.items():
        p, q = example["parameters"], example["dJAX"]
        H = hamiltonian(*p)
        print("\n" + name)
        print("Five parameters:", p)
        print(H)
        print("All eigenvalues:", np.linalg.eigvalsh(H))
        print("Principal gaps (dJAX=0):", principal_frequencies(H))
        print(
            "Principal gaps (with fitted dJAX), even:",
            principal_frequencies(hamiltonian(*p, dJAX=q)),
        )
        print(
            "Principal gaps (with fitted dJAX), odd:",
            principal_frequencies_odd(hamiltonian(*p, dJAX=q, parity="odd")),
        )
    if args.samples:
        if args.csv is None:
            parser.error("--samples requires an explicit --csv path")
        _write_csv(args.csv, make_dataset(args.samples))
        print(f"\nSaved {args.samples} training samples to {args.csv}")


if __name__ == "__main__":
    main()
