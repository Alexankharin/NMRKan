"""Locations for immutable reference artifacts and new molecular experiments."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = PROJECT_ROOT / "results/reference/molecules"
OUTPUT_ROOT = Path(
    os.environ.get("NMRKAN_MOLECULE_OUTPUT", PROJECT_ROOT / "outputs/molecules")
).resolve()
if OUTPUT_ROOT == REFERENCE_ROOT or REFERENCE_ROOT in OUTPUT_ROOT.parents:
    raise ValueError("Choose an output directory outside results/reference.")
