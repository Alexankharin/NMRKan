# Validation of the repository cleanup

Validated on 8 November 2026 using Linux x86-64, Python 3.14.4, CPU PyTorch 2.15.0, NumPy 2.5.4, and SymPy 1.14.0. Exact package versions are recorded in [the environment record](../results/validation/environment.json). Historical molecular training environments remain in their original configurations; they were not rewritten to match this environment.

## Completed checks

| Check | Result |
|---|---|
| `python -m pytest -q` | 39 tests pass, including short three-site and four-site training and extraction |
| `python -m experiments.check_notebooks` | All five numbered notebooks execute from fresh kernels; executed copies are under ignored `outputs/executed_notebooks/` |
| `python -m experiments.verify_artifacts` | All 148 supplied molecular reference files match their original SHA-256 values |
| Relocation audit | All 216 moved files accounted for; raw data, saved results and archived content preserved |
| Three-site full-size numerical command | 100,000 generated targets; quadratic/radial OLS; perturbative and rounded-formula checks; figures and numerical exports |
| Four-site primary numerical command | 200 x 200 tracked/interpolated grid, 10,000 physical samples; OLS and rounded-formula metrics |
| Four-site verification numerical command | 10,000 uniform ratios, direct diagonalization; OLS and rounded-formula metrics |
| Molecular report rebuild | DSS analytical, DSS trained comparison, and homotaurine reports rebuilt; all 12 report images embedded and all template values filled |
| Molecular training smoke run | DSS/local, 256 samples, 10 epochs, seed 42; dataset, checkpoint, formulas, OLS and comparison outputs created |
| `python -m pip check` | No broken dependency requirements |

The test suite independently constructs the full four-spin operator and checks the supplied four-site block. The existing six-spin projection tests also validate the molecular Hamiltonian. Saved molecular checkpoints are reloaded and their symbolic formulas and held-out errors are checked. A pre-existing dimensional-analysis crash on a dimensionless sum was fixed and covered by an assertion.

## Measured numerical results

Full configurations and metrics are in [paper_baselines.json](../results/validation/paper_baselines.json).

| Calculation | Recomputed dimensionless RMSE | Manuscript comparison |
|---|---:|---|
| Three-site perturbation, 101 x 101 local grid | 0.00004282601 | Matches 4.28e-5 at the quoted precision |
| Rounded Eq. (12), 121 x 121 grid | 0.01903523 | Matches 1.90e-2 at the quoted precision |
| Three-site full-sample quadratic OLS | 0.01613114 | Slight difference from 1.62e-2 |
| Three-site radial OLS, validation | 0.01402045 | Matches 1.40e-2 at the quoted precision |
| Four-site OLS, uniform ratios | 0.01822717 | Does not reproduce the reported 7.02e-3 |
| Rounded Eq. (13), uniform ratios | 0.04139222 | Evaluation on the stated independent verification distribution |
| Four-site OLS, physical/interpolated sampling | 0.00683418 | Different sampling protocol |
| Rounded Eq. (13), physical/interpolated sampling | 0.00702377 | Different sampling protocol and different predictor |

These results expose a manuscript/protocol discrepancy; they are not substituted into the manuscript PDF. The original independent verification implementation/data are needed to identify its cause. The three-site rounded expressions and local perturbative calculation agree with the reported values at the displayed precision.

## Subsequent full rerun

The later [full rerun](../results/reproduced/2026-11-08/README.md) completes all four paper neural protocols and reruns the maintained notebooks. Its comparison and exact artifacts are separate from this initial cleanup validation record. A checkpoint serialization fix now stores the PyTorch version as a plain string, so newly saved checkpoints reload with `weights_only=True`.

## Limits of the initial cleanup validation

At the initial cleanup checkpoint, full 1000/5000-epoch neural training and the original architecture search had not been rerun. Full neural runs are now available at the link above; the original architecture-search artifacts remain absent. The original main-paper neural checkpoints/search table were absent; new short training checks establish that the workflow and symbolic extraction execute, not that historical fitted coefficients have been recovered. Molecular tests validate the supplied twenty checkpoints rather than retraining their four five-seed experiments.

The added GitHub Actions workflow runs artifact verification, the test suite, all active notebooks, and molecular report rebuilding. This record reports local execution; remote CI results depend on the subsequent GitHub run. Archived exploratory notebooks are preserved for provenance and are excluded from active notebook validation.
