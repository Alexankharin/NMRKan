# Executable notebooks

Install the repository using the root README, then run `jupyter lab notebooks` from the root. Each notebook runs from top to bottom with the same environment; none depends on another notebook's kernel state.

| Order | Notebook | What it demonstrates |
|---|---|---|
| 00 | [Model and Hamiltonians](00_model_and_hamiltonians.ipynb) | Exact multiplication with ASN layers; three/four-site Hamiltonians and conventions |
| 01 | [Three-site paper results](01_three_site_paper.ipynb) | Perturbation errors, published formulas, symmetry, quadratic and radial OLS |
| 02 | [Four-site paper results](02_four_site_paper.ipynb) | Direct eigenvalues, rounded expressions and a matched quadratic baseline |
| 03 | [Real molecules](03_real_molecules.ipynb) | DSS/homotaurine physics, saved model metrics, checkpoint/formula equivalence |
| 04 | [Training walkthrough](04_training_walkthrough.ipynb) | Completed 1000-epoch training, checkpoint reload, symbolic extraction and optional full retraining |

The notebooks in this directory include executed cells, printed results, tables, and inline plots, so they can be read directly on GitHub or in Jupyter. `python -m experiments.check_notebooks --in-place` reruns all five and saves their outputs here after every notebook succeeds; it also writes copies under `outputs/executed_notebooks/`. Omit `--in-place` to validate without updating the checked-in files. Keep outputs when committing these notebooks.

The [saved execution snapshot](../results/reproduced/2026-11-08/README.md) records the completed full training runs they display. Notebook 04 can optionally repeat the full three-site optimization. To repeat all four protocols and rerun the notebooks, use the commands in the reproduction guide.

Earlier notebooks and their outputs are preserved in `archive/notebooks/`. Static historical molecular report notebooks are under `results/reference/molecules/reports/`; their old commands are provenance, and the active report builders generate updated commands.

To execute the generated molecular report notebooks as well, first rebuild them with `python -m experiments.molecules --from-reference`, then run `python -m experiments.check_notebooks --molecular-reports outputs/molecules`.
