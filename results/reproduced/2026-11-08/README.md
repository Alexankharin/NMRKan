# Full paper and notebook rerun — 8 November 2026

All five maintained notebooks and all three regenerated molecular report notebooks executed successfully in fresh kernels. All four paper neural protocols completed their full training budgets. The rerun recovers similar primary-benchmark behavior, but does **not** reproduce every numerical claim in [manuscript v24](../../../paper/manuscript_v24.pdf).

## Read the executed notebooks

These copies include the actual outputs and inline figures. The main notebooks in [`notebooks/`](../../../notebooks/README.md) also retain their completed execution outputs, so readers can view results directly in either location.

| Executed notebook | Contents |
|---|---|
| [00 — model and Hamiltonians](notebooks/00_model_and_hamiltonians.ipynb) | ASN multiplication construction and Hamiltonian conventions |
| [01 — three-site paper results](notebooks/01_three_site_paper.ipynb) | Perturbation theory, published formulas, OLS, and reloaded full ASN |
| [02 — four-site paper results](notebooks/02_four_site_paper.ipynb) | Both sampling protocols, full trained models, and the verification discrepancy |
| [03 — real molecules](notebooks/03_real_molecules.ipynb) | Reference checkpoints, formula equivalence, and physical predictions |
| [04 — full training walkthrough](notebooks/04_training_walkthrough.ipynb) | Completed 1000-epoch history, symbolic extraction, and optional full retraining |
| [DSS analytical report](molecular_reports/dss_analysis.ipynb) | Recomputed physical energies, frequencies, Taylor coefficients, and transitions |
| [DSS ASN comparison](molecular_reports/dss_asn_comparison.ipynb) | Rebuilt reference-model evaluation and extracted expressions |
| [Homotaurine ASN comparison](molecular_reports/homotaurine_asn_comparison.ipynb) | Rebuilt reference-model evaluation and extracted expressions |

The molecular reports also have self-contained HTML versions in `molecular_reports/`. Their twenty reference checkpoints were validated and reused; these molecular models were not retrained. Archived exploratory notebooks remain historical records: some contain missing private inputs or malformed notebook data and are outside the supported reproduction route.

## Training and comparison

| Completed run | Samples | Epochs | Sampling |
|---|---:|---:|---|
| [Three-site primary](paper/three_site/) | 100,000 | 1000 | Uniform ratios, separate coordinate draws |
| [Three-site verification](paper/three_site_verification/) | 20,000 | 400 | Uniform ratios, separate coordinate draws |
| [Four-site primary](paper/four_site/) | 10,000 | 5000 | Physical couplings, tracked 200 × 200 interpolation grid |
| [Four-site verification](paper/four_site_verification/) | 10,000 | 5000 | Uniform ratios, direct diagonalization |

All runs use recorded data, split, and model seeds of 42, CPU float64, and the final epoch checkpoint. Each run directory contains its configuration, sampled dataset and split indices, history, checkpoint, raw and processed formulas, OLS coefficients, metrics, and PDF/PNG figures. Training logs are in [`paper/logs/`](paper/logs/).

| Dimensionless RMSE and evaluation subset | Paper | Rerun |
|---|---:|---:|
| Three-site primary ASN, complete sample | 0.0179 | 0.01775957 |
| Four-site primary ASN, validation | 0.0079 | 0.00683380 |
| Four-site verification ASN, validation | 0.00753 | 0.01991641 |
| Four-site verification quadratic OLS, validation | 0.00702 | 0.01822717 |

None of these four numbers matches at the paper's displayed precision. The reduced three-site verification additionally gives validation RMSE 0.01779838. All extracted raw neural expressions are quadratic, and every activation-selection vector has support size one. Reloaded neural predictions agree with the raw formulas to at most 1.5e-14 absolute error. The full audit, per-branch R², processed-formula metrics, and comparison are in [`comparison.json`](comparison.json).

The original main-paper CUDA checkpoints and model initialization seeds were not supplied. This prevents reconstructing the historical optimization exactly. More substantially, the stated uniform-ratio four-site protocol disagrees with the reported OLS error independently of neural initialization. The [reproduction guide](../../../docs/reproducing_the_paper.md#numerical-discrepancy-requiring-manuscript-review) records the separate physical-coupling and uniform-ratio calculations. Resolving that discrepancy requires the historical verification code/data or a manuscript correction. No seeds or sampling distributions were selected to force agreement.

The deterministic three-site local perturbation RMSE (4.28260081e-5) and rounded Eq. (12) grid RMSE (0.01903523) do match the manuscript's displayed precision.

## Repeat or inspect

From the repository root, after the [installation](../../../README.md#install):

```bash
# Reload and audit these four complete checkpoints.
python -m experiments.reproduction

# Execute the five maintained notebooks against the saved full runs.
python -m experiments.check_notebooks

# Refresh the executed outputs in the main notebooks before committing them.
python -m experiments.check_notebooks --in-place

# Regenerate and execute the molecular reports as well.
python -m experiments.molecules --from-reference
python -m experiments.check_notebooks --molecular-reports outputs/molecules

# Repeat all four training protocols into a fresh folder, then inspect them.
python -m experiments.reproduction --train --root outputs/full_paper_repeat --summary outputs/full_paper_repeat/comparison.json
python -m experiments.check_notebooks --paper-runs outputs/full_paper_repeat
```

## Provenance and checks

Training started from cleanup commit `80e97f10dfa21bbffa6acb5933465b4e89ab7c47`. [`paper/runner_at_training.py`](paper/runner_at_training.py) preserves the exact training runner; its SHA-256 matches the value in all four run configurations. The active runner subsequently received a checkpoint-metadata compatibility fix and a plotting-backend fix. The checkpoint repair converted only the stored PyTorch version to a plain string; tensor values were preserved. Before/after hashes are recorded in [`paper/checkpoint_metadata_repair.json`](paper/checkpoint_metadata_repair.json).

[`environment.json`](environment.json) records package versions and execution settings. [`notebook_execution.json`](notebook_execution.json) records every executed notebook's code-cell and error counts. Relative documentation links in the saved numbered notebooks are adjusted for the snapshot location; code and execution outputs are preserved. [`manifest.json`](manifest.json) records SHA-256 checksums for this snapshot's files, excluding itself.

Validation completed with 39 passing tests, all eight supported notebooks executed without cell errors, checkpoint/formula reload checks for all four paper runs, and unchanged checksums for all 148 molecular reference files.
