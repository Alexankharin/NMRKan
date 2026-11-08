"""Rebuild the molecular reports, optionally using the supplied trained models."""

import argparse
import shutil

from .paths import OUTPUT_ROOT, REFERENCE_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-reference",
        action="store_true",
        help="Copy supplied results to outputs before rebuilding reports; no training.",
    )
    args = parser.parse_args()
    if args.from_reference:
        for name in ("analytical", "dss", "homotaurine"):
            destination = OUTPUT_ROOT / name
            if destination.exists():
                raise FileExistsError(
                    f"{destination} already exists; choose a fresh NMRKAN_MOLECULE_OUTPUT."
                )
        for name in ("analytical", "dss", "homotaurine"):
            shutil.copytree(REFERENCE_ROOT / name, OUTPUT_ROOT / name)
    from nmrkan.physics.real_molecules import build_results
    from . import analytical_report, dss_report, homotaurine_report

    build_results(OUTPUT_ROOT / "analytical", figures=True)
    analytical_report.build()
    dss_report.main()
    homotaurine_report.main()
    print(f"Reports and figures: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
