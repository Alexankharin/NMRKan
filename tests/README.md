# Validation suite

Run `python -m pytest` from the root after installing `.[reproduce]`.

- `test_paper.py`: independent signed-energy/branch checks, published perturbation and rounded-formula errors, four-site symmetry, disjoint splits and a short ASN training/extraction run.
- `molecules/`: six-spin projection, analytical/numerical identities, limiting cases, physical spectral checks, dataset separation, selected-checkpoint reloads and learned-formula agreement for DSS and homotaurine.
- Remaining modules: existing dimensional-analysis, conversion and data-interface checks.

The reference-artifact tests use checked-in local files and require no training download or external account. Long scientific training is deliberately separate. Run `python -m experiments.check_notebooks` for all five active notebooks and `python -m experiments.verify_artifacts` for preserved artifact checksums.
