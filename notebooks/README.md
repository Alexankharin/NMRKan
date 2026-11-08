# Executable notebooks

Install the repository using the root README, then run `jupyter lab notebooks` from the root. Each notebook runs from top to bottom with the same environment; none depends on another notebook's kernel state.

| Order | Notebook | What it demonstrates |
|---|---|---|
| 00 | [Model and Hamiltonians](00_model_and_hamiltonians.ipynb) | Exact multiplication with ASN layers; three/four-site Hamiltonians and conventions |
| 01 | [Three-site paper results](01_three_site_paper.ipynb) | Perturbation errors, published formulas, symmetry, quadratic and radial OLS |
| 02 | [Four-site paper results](02_four_site_paper.ipynb) | Direct eigenvalues, rounded expressions and a matched quadratic baseline |
| 03 | [Real molecules](03_real_molecules.ipynb) | DSS/homotaurine physics, saved model metrics, checkpoint/formula equivalence |
| 04 | [Training walkthrough](04_training_walkthrough.ipynb) | Ten-epoch smoke training, checkpointing, symbolic extraction and processing |

`python -m experiments.check_notebooks` executes all five and writes executed copies under `outputs/executed_notebooks/`. Committed notebooks have no execution outputs. Full scientific training is an explicit CLI operation documented within the notebooks; the short walkthrough does not reproduce paper accuracy.

Earlier notebooks and their outputs are preserved in `archive/notebooks/`. Static historical molecular report notebooks are under `results/reference/molecules/reports/`; their old commands are provenance, and the active report builders generate updated commands.
