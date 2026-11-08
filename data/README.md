# Input data and historical sources

`raw/2024/` and `raw/2025/` preserve the original tables, spreadsheets, Mathematica notebooks, plotting script and source ZIP. Files are retained unchanged, including historical duplicate exports. Their provenance is the original repository; no new experimental measurements are introduced here.

The supported paper workflows generate synthetic labels directly from their Hamiltonians, so these historical tables are not prerequisites. Molecular datasets, including exact split indices, live with their trained artifacts in `results/reference/molecules/{dss,homotaurine}/{local,broad}/dataset.npz`.

Generated data belongs with a run in `outputs/` (or a local cache under `data/cache/`), not in `raw/`. Paths inside archived notebooks refer to the old layout; consult `archive/relocation.json` to locate their inputs.
