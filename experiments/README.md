# Experiment entry points

Run these as modules from the installed checkout:

| Module | Purpose |
|---|---|
| `experiments.paper` | Main paper benchmarks, explicit JSON configs, optional neural training |
| `experiments.molecules` | Rebuild molecular reports; `--from-reference` copies the supplied artifacts first |
| `experiments.molecules.train` | Train DSS/homotaurine local/broad five-seed models |
| `experiments.molecules.analytical_report` | Build the analytical DSS report after analytical outputs exist |
| `experiments.molecules.dss_report` | Evaluate DSS saved models, generate plots, HTML/notebook and TeX insertion |
| `experiments.molecules.homotaurine_report` | The corresponding homotaurine analysis/report |
| `experiments.reproduction` | Retrain all four full protocols or reload the completed runs and compare manuscript metrics |
| `experiments.check_notebooks` | Execute the five supported notebooks and, with `--molecular-reports`, the three rebuilt report notebooks |
| `experiments.verify_artifacts` | Verify preserved reference checksums |
| `experiments.exploratory.train` | Earlier generic configurable trainer; not the paper protocol |

See [the reproduction guide](../docs/reproducing_the_paper.md) for inputs, outputs, and scientific distinctions. New artifacts belong under `outputs/`, never beside source modules.
