# Preserved reference results

`molecules/` contains the complete supplied real-molecule analysis, relocated from the previously untracked `ANSWER/` folder. Scientific artifacts are preserved byte for byte. These are reference inputs; new runs write to `outputs/`.

| Location | Contents |
|---|---|
| `molecules/analytical/` | Exact/reference spectra, sections, comparison tables and figures |
| `molecules/dss/{local,broad}/` | Dataset and splits, configs, five checkpoints, histories, neural/OLS predictions, formulas and comparisons |
| `molecules/homotaurine/{local,broad}/` | The corresponding homotaurine trained experiments |
| `molecules/dss/`, `molecules/homotaurine/` | Cross-domain figures, analytical additions and manuscript TeX insertions |
| `molecules/reports/` | Supplied static PDF, HTML and report-notebook snapshots |

`manifest.json` records each file's SHA-256, size and original path. Run `python -m experiments.verify_artifacts` to check it. The static reports preserve historical commands and paths; regenerate current HTML/notebooks with `python -m experiments.molecules --from-reference`.

The broad DSS and homotaurine datasets are intentionally identical. Twenty checkpoint files are retained across the four molecule/domain combinations. Model-source hashes in original configs still refer to the unchanged `nmrkan/models.py`. Original numerical training versions remain recorded in those configs. No original main-paper neural checkpoints were supplied.
