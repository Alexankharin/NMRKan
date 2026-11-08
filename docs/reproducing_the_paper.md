# Reproducing the paper

The reference manuscript is [v24](../paper/manuscript_v24.pdf), dated 25 August 2026. Run commands from the repository root after the installation in the [README](../README.md). Numerical outputs are dimensionless unless an artifact explicitly ends in `_hz`.

## Result map

| Manuscript result | Executable route | Output or evidence | Reproduction scope |
|---|---|---|---|
| Fig. 1 and Appendix A, activation selection and multiplication | [00](../notebooks/00_model_and_hamiltonians.ipynb) | Exact multiplication check with the actual ASN layers | Mathematical construction; original diagram artwork is in the PDF |
| Fig. 2 and Appendix B, Hamiltonian blocks | [00](../notebooks/00_model_and_hamiltonians.ipynb), `nmrkan/physics/three_ch2.py`, `data_generation/eigenvalues_4CH2.py` | Matrices, trace, scale, and six-spin projection checks | Recomputed from Hamiltonians |
| Eqs. (9–11), Fig. 3, Appendix D | [01](../notebooks/01_three_site_paper.ipynb); three-site command below | `fig03_sections.*`, `figure_data.npz`, `perturbation_local_grid` in `metrics.json` | Local RMSE 4.28e-5 and maximum error 2.64e-4 can be checked independently |
| Eq. (12), Fig. 4, Sec. V A | [01](../notebooks/01_three_site_paper.ipynb); three-site command | `fig04_surfaces.*`, grid and uniform-section errors | Default plots evaluate the published rounded coefficients; `--train` plots a fresh ASN and labels it accordingly |
| Sec. V A, original RMSE 1.79e-2 and per-branch R² | Three-site command with `--train` | `model.pt`, `history.csv`, `learned_formulas.json` | Original checkpoint absent; the completed CPU rerun gives 0.01775957 and is saved separately |
| Sec. V B and VIII D, quadratic and radial OLS | [01](../notebooks/01_three_site_paper.ipynb); three-site command | `ols.npz`, `radial_coefficients.npy`, full-sample, validation, and local-grid metrics | Same 100,000 inputs and 80/20 split; full-sample fit is explicitly separated from validation |
| Eq. (13), Fig. 5, Sec. V C | [02](../notebooks/02_four_site_paper.ipynb); four-site primary command | `fig05_branches.*`, `figure_data.npz` | Rounded equations available; original neural checkpoint absent |
| Sec. V D, Eq. (14), VIII E, four-site OLS and fresh ASN verification | Four-site verification command, optionally `--train` | `ols.npz`, `metrics.json`, checkpoint and extracted formulas when trained | Uses direct diagonalization and uniform ratios; the separately written historical verification implementation was not supplied |
| VIII E, reduced three-site verification | `three_site_verification.json` with `--train` | 20,000 samples, 400 epochs, fresh metrics and support sizes | Completed 400-epoch verification run |
| Appendix C, Pareto architecture selection | `archive/notebooks/2d_task.ipynb`, `archive/scripts/` | Historical exploratory code and manuscript description | Search table, full candidate list, and selected-run artifacts are absent; the selected architectures are configurable, but the original Pareto front cannot be recovered |
| Sec. VI, repeated initializations | Molecular five-seed artifacts; paper configs expose `model_seed` | Molecular `training_summaries.json`; fresh paper runs can use copied configs | Original benchmark five-seed aggregates were not supplied |
| Real-molecule extension requested by editor | [03](../notebooks/03_real_molecules.ipynb), molecular commands below | DSS/homotaurine reference artifacts and rebuilt reports | Extension to v24; not already part of its numbered figures |

## Completed notebook and full training rerun

The [8 November 2026 rerun](../results/reproduced/2026-11-08/README.md) contains completed primary and verification training for both spin chains: 1000, 400, 5000, and 5000 epochs respectively. All checkpoints, exact sampled datasets/splits, training histories, formulas and figures are retained. The numbered notebooks display these full results alongside the published rounded equations. The original unrecorded CUDA model initialization is not reconstructed by these CPU runs.

To repeat every full paper protocol into a new folder and execute all maintained notebooks against those models:

```bash
python -m experiments.reproduction --train --root outputs/full_paper_repeat --summary outputs/full_paper_repeat/comparison.json
python -m experiments.check_notebooks --paper-runs outputs/full_paper_repeat
```

To reload and verify the completed runs without training again:

```bash
python -m experiments.reproduction
python -m experiments.check_notebooks
```

The complete-sample three-site ASN RMSE is 0.01775957 (paper: 0.0179). Primary four-site validation RMSE is 0.00683380 (paper: 0.0079). Uniform-ratio verification ASN RMSE is 0.01991641 (paper: 0.00753). None is an exact match at the paper's displayed precision. Full training therefore recovers similar primary-benchmark behavior but leaves the independent four-site protocol discrepancy unresolved. The comparison keeps each metric's sampling distribution and evaluation subset explicit.

The main files in `notebooks/` include their completed printed results, tables, and inline figures. To refresh those checked-in outputs, use `python -m experiments.check_notebooks --in-place`. Without that flag, execution writes separate copies under `outputs/`.

## Three-site benchmark

```bash
python -m experiments.paper --config configs/paper/three_site.json
python -m experiments.paper --config configs/paper/three_site.json --train
python -m experiments.paper --config configs/paper/three_site_verification.json --train
```

The first command generates 100,000 input pairs with separate NumPy `RandomState(42)` draws for each coordinate, matching the original notebook. Both ratios range from 0 to 2. A PyTorch permutation with seed 42 defines the 80/20 train/validation split. OLS fits use only training rows for validation metrics. A separate complete-sample fit reproduces the coefficient-comparison protocol in Sec. VIII D.

The primary ASN uses `(2,16,4)`, Adam at 0.001, MAE, batches of 4096, and 1000 epochs. Softmax plus 0.1 times the L0.5 penalty is used for the first 90%; sparsemax without the penalty is used for the last 10%. The new runner records model seed 42 and the final epoch as its checkpoint convention. Original initialization/checkpoint-selection details were not fully preserved; do not expect exact historical coefficients.

All branches are fitted simultaneously. Sign-reflected plots evaluate the first-quadrant model at `abs(r1), abs(r2)`. Raw extracted expressions are checked against network evaluation before symmetry averaging, thresholding at 0.001, and rounding to three decimals. The saved report distinguishes neural errors from post-processing errors. Training is unconstrained and can retain higher-degree terms; it is not forced to return Eq. (12).

## Four-site benchmark

```bash
# Original sampling protocol: physical couplings, tracked/interpolated 200 x 200 grid.
python -m experiments.paper --config configs/paper/four_site.json
python -m experiments.paper --config configs/paper/four_site.json --train

# Independent protocol: 10,000 uniform ratios in [0,3], direct diagonalization.
python -m experiments.paper --config configs/paper/four_site_verification.json
python -m experiments.paper --config configs/paper/four_site_verification.json --train
```

Both neural protocols use `(1,10,8)`, MSE, Adam at 0.001, batches of 8000, and 5000 epochs. The primary physical ranges are `J in [-32,-5]` and `deltaJ in [-15,-0.1]` Hz. Dividing by the signed J and sorting gives descending dimensionless labels. The derived ratio distribution is nonuniform; it must not be confused with the uniform-ratio verification dataset.

`--quick` reduces a workflow to 256 samples and 10 epochs (and a 20 x 20 interpolation grid where applicable). It is a wiring check, not evidence for reported accuracy. [04](../notebooks/04_training_walkthrough.ipynb) now inspects the completed full training run and supports optional retraining.

## Numerical discrepancy requiring manuscript review

The supplied eight-dimensional Hamiltonian and the stated **uniform-ratio** verification protocol produce an OLS validation RMSE of **0.01822717**, rather than the **0.00702** reported in Sec. V D. Evaluating rounded Eq. (13) on that same split gives **0.04139222**. The original physical-coupling/interpolation sampling gives OLS validation RMSE **0.00683418**, and rounded Eq. (13) gives **0.00702377** on that split.

These are different datasets and metrics. The new commands preserve each stated sampling protocol instead of changing it to match the quoted value. The four-site matrix was checked against Fig. 2 and an independently constructed full effective-spin operator. Resolving the reported verification value requires the historical verification code/data or a manuscript correction. The numerical baseline discrepancy does not require neural retraining to expose.

Smaller differences also remain in three-site OLS: this reconstruction gives complete-sample RMSE 0.01613114 (manuscript 0.0162), local quadratic RMSE 0.01718853 (manuscript 0.0174), and local radial RMSE 0.00652027 (manuscript 0.00658). The deterministic perturbation-grid and rounded-formula-grid checks agree with the manuscript's displayed precision. See [validation](validation.md) for the measured outputs.

## Molecular extension

To inspect the supplied results without retraining:

```bash
python -m pytest tests/molecules
python -m experiments.molecules --from-reference
python -m experiments.check_notebooks --molecular-reports outputs/molecules
```

The second command copies artifacts into `outputs/molecules/`, regenerates analytical calculations and figures, and writes self-contained HTML reports and report notebooks there. The third executes all five numbered notebooks and all three generated report notebooks. The original PDF reports are preserved in `results/reference/molecules/reports/`; generating new PDFs requires printing the rebuilt HTML to PDF. The report builders themselves produce HTML, notebooks, figures, and TeX manuscript insertions.

To generate independent datasets and train every seed from scratch, choose a fresh directory:

```bash
export NMRKAN_MOLECULE_OUTPUT=outputs/molecules_retrained
python -m experiments.molecules.train --molecule dss --domain local
python -m experiments.molecules.train --molecule dss --domain broad
python -m experiments.molecules.train --molecule homotaurine --domain local
python -m experiments.molecules.train --molecule homotaurine --domain broad
python -m experiments.molecules
```

Defaults are 100,000 physical samples per domain, disjoint 80/10/10 train/validation/test splits, seeds 42–46, `(4,16,4)`, 1000 epochs, and batches of 4096. Conditioning uses training rows only. Model selection uses validation error during the sparsemax phase; test errors are evaluated afterwards. `config.json` records bounds, conditioning, versions and hashes. Each domain also retains all five checkpoints, histories, split indices, held-out predictions, OLS degree-2/4 baselines, and learned formulas.

The DSS and homotaurine broad-domain datasets are identical by design; they are the same five-parameter benchmark with different reference cross-sections. They must not be counted as independent validation datasets. The local boxes differ, and neither is guaranteed to be contained in the broad box.

## Energy and frequency conventions

- Paper: `r_i = deltaJ_i/J`, `epsilon = E/J`, with descending `epsilon`. With physical `J < 0`, ascending physical energies map to descending dimensionless energies.
- Molecules: `s = -JMM > 0`, `E_ref = (-3 JAA + JMM - 3 JXX)/4`, `lambda = (E-E_ref)/s`, ascending `lambda`. Inputs are `((JAA-JMM)/s, (JXX-JMM)/s, deltaJAM/s, deltaJMX/s)`.
- `deltaJ` is the full coupling difference; off-diagonal matrix entries use `deltaJ/2`. Couplings and spectral frequencies are in Hz, not rad/s.
- Molecular principal lines use the upper three even-parity physical energies. Fixed branch-index differences preserve ordering violations in learned predictions. General spectral visibility needs eigenvectors, an initial state, and an observable.
- The molecular Hamiltonian is exact within its stated secular model. Optional non-nearest-neighbour coupling and illustrative line intensities are analyzed separately; they are not a full experimental spectrum fit.

## Outputs and provenance

Every paper run writes `config.json`, `dataset.npz` with saved split indices, `ols.npz`, `metrics.json`, numerical figure data, and PNG/PDF figures. Neural runs additionally write `model.pt`, `history.csv`, and `learned_formulas.json`. Reusing a completed directory is rejected; provide `--output` for a new run.

The notebooks run without private files or the historical Mathematica environment. Archived notebooks may contain obsolete imports and missing external input references; they are preserved as provenance, not included in the supported execution path. [Validation](validation.md) records the checks performed during this reorganization.
