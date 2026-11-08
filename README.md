# NMRKan

Code and reproducibility materials for **Interpretable Activation-Selection Neural Networks for Symbolic Regression of Parameter-Dependent Hamiltonian Eigenvalues**, Alexander Yu. Kharin and Kirill F. Sheberstov, manuscript v24 (25 August 2026).

The model is an **activation-selection network (ASN)**: each input to a layer selects from analytical functions, followed by a linear transformation. The Python class retains its historical name, `KharKAN`. The paper uses the library `{0, x, x²}` to approximate dimensionless spin-chain eigenvalues.

**Start with [the result-by-result reproduction guide](docs/reproducing_the_paper.md)** or open the [numbered notebooks](notebooks/README.md). The [reviewer response map](docs/reviewer_response.md) explains the DSS and homotaurine extension and the remaining scientific work.

## Install

Use Python 3.10 or newer, from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU is sufficient for every supported workflow.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[reproduce]'
```

The editable installation makes imports work from notebooks without changing `sys.path`. `requirements.txt` contains the core dependencies; the `reproduce` extra adds notebooks, tests, and report rendering. Recorded molecular training environments are in each saved `config.json`; a newer environment can validate the artifacts without reproducing neural optimization bit for bit.

## Reproduce results

Run these commands from the repository root after installation:

```bash
# Exact eigenvalues, perturbation theory, published rounded formulas, OLS baselines,
# and figures. Neither command trains a neural network.
python -m experiments.paper --config configs/paper/three_site.json
python -m experiments.paper --config configs/paper/four_site_verification.json

# Rebuild the real-molecule reports using the supplied trained models.
python -m experiments.molecules --from-reference

# Validate physics, reference checkpoints, formulas, and a short training run.
python -m pytest

# Explore the explanations and calculations interactively.
jupyter lab notebooks
```

New files go to `outputs/`, which is ignored by Git. Existing run directories are protected: use `--output outputs/paper/another_run` for a repeated paper calculation, or set `NMRKAN_MOLECULE_OUTPUT=outputs/molecules_another_run` for molecular reports. Reference artifacts under `results/reference/` are retained separately.

For fresh ASN training:

```bash
# Short workflow check; these metrics are not paper results.
python -m experiments.paper --config configs/paper/three_site.json --train --quick

# Full manuscript training protocols; expect substantially longer runs.
python -m experiments.paper --config configs/paper/three_site.json --train
python -m experiments.paper --config configs/paper/four_site.json --train
```

**Reproducibility finding:** the stated four-site uniform-ratio OLS protocol does not reproduce the manuscript’s quoted error. The [guide records both sampling protocols and the measured discrepancy](docs/reproducing_the_paper.md#numerical-discrepancy-requiring-manuscript-review).

The original three-site and four-site neural checkpoints and architecture-search table were not supplied. Fresh runs save their own seeds, splits, metrics, checkpoints, and extracted formulas; they are not represented as the historical runs. Published rounded equations can be evaluated directly. The molecular extension includes all supplied datasets and twenty saved checkpoints.

## Repository layout

| Folder | Contents |
|---|---|
| [`paper/`](paper/README.md) | Manuscript, editorial comments, and molecular Hamiltonian note |
| [`docs/`](docs/README.md) | Reproduction guide, reviewer response, conventions, and contribution rules |
| [`notebooks/`](notebooks/README.md) | Numbered, executable explanations of the paper and molecular extension |
| [`experiments/`](experiments/README.md) | Paper CLI, molecular training/report CLIs, and earlier exploratory interface |
| [`configs/`](configs/README.md) | Named manuscript protocols; older YAML settings under `exploratory/` |
| [`nmrkan/`](nmrkan/README.md) | ASN model, physics, training, symbolic extraction, and utilities |
| [`data_generation/`](data_generation/README.md) | Synthetic spin-chain data and original interpolation routines |
| [`data/`](data/README.md) | Historical input tables and Mathematica source, under `raw/` |
| [`results/reference/`](results/reference/README.md) | Preserved molecular results, checkpoints, reports, and checksums |
| [`tests/`](tests/README.md) | Physics, trained-artifact, paper-protocol, and dimensional-analysis checks |
| [`archive/`](archive/README.md) | Earlier notebooks, scripts, development notes, logs, and relocation record |
| `outputs/` | Locally generated runs; created by the commands above |

## Scientific conventions

The paper orders dimensionless eigenvalues `E/J` from highest to lowest. Physical `J < 0`, so physical energies have the reverse order. Molecular experiments instead use the positive scale `s = -JMM` and an energy offset, and sort their dimensionless eigenvalues in ascending order. These conventions are documented in [the reproduction guide](docs/reproducing_the_paper.md#energy-and-frequency-conventions).

The molecular spectra are model predictions based on supplied fitted coupling rows. Illustrative transition amplitudes depend on the chosen state preparation and observable; they are not newly measured spectra. OLS is a required baseline, and the existing comparisons do not establish an ASN accuracy advantage over a suitable fixed basis.

## Contributing and citation

See [contribution and maintenance rules](docs/contributing.md). Cite the manuscript by the title and authors above; no DOI or published journal citation is supplied. The repository does not currently contain a license grant.
