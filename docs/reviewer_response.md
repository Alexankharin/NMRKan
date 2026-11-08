# Reviewer response and remaining work

The supplied [editorial letter](../paper/editorial_comments.txt) requests (a) NMR predictions for real materials or molecules and (b) stronger literature context and comparison with state-of-the-art models. The repository separates evidence already available from work still required for resubmission.

| Request | Available evidence and how to inspect it | Remaining work |
|---|---|---|
| Real-molecule predictions | DSS and homotaurine five-parameter Hamiltonians; analytical reference; actually trained four-input ASNs; local/broad domains and five initializations. Run [notebook 03](../notebooks/03_real_molecules.ipynb) or `python -m experiments.molecules --from-reference`. | Integrate the generated manuscript additions into the paper and resolve experimental provenance/conditions. No newly measured spectrum is supplied. |
| Comparison with alternatives | Exact diagonalization, exact analytical cubic, reduced 3x3 models, second-order perturbation theory, and OLS degree-2/4 comparisons use explicit domains and metrics. Paper commands also include symmetric quadratic and radial-feature OLS. | These are analytical and fixed-basis baselines. Matched benchmarks against published symbolic-regression systems have not been supplied; do not label this requirement complete. |
| Literature context | The v24 introduction discusses EQL, KANs, symbolic regression and perturbation theory; original sources remain in the manuscript. | A broader, critically connected literature discussion and justified selection of external comparison methods remain manuscript work. This cleanup does not add unverified citations or claim state-of-the-art performance. |

## Molecular evidence

Supplied fitted coupling rows are defined in `nmrkan/physics/three_ch2.py::EXAMPLES`. DSS is labeled as a 298 K fit from a 31 July 2026 email; homotaurine is from a 15 August 2026 email with unresolved temperature labeling. Those emails and original experimental data are not present. The supplied [Hamiltonian note](../paper/molecular_hamiltonian.pdf) and preserved reports describe the assumptions.

For each molecule, `results/reference/molecules/<molecule>/{local,broad}/` contains training settings, numerical targets, splits, all five checkpoints, histories, model comparisons, and full symbolic expressions. `learned_formulas.json` contains learned expressions; the analytical cubic and local Taylor expansions are separate references. The tests reload selected checkpoints, compare network and symbolic evaluations, recompute saved error metrics, and verify split separation.

Use the generated `dss_manuscript_addition.tex` and `homotaurine_manuscript_addition.tex` as draft insertions. Their results are model predictions conditional on the supplied coupling rows, not independent molecular measurements. Illustrative spectra assume a specified preparation and observable. The reports also quantify non-nearest-neighbour coupling and reduced-model errors.

## Resubmission checklist

- Integrate the molecular methods, figures, metrics, and domain limitations into the manuscript.
- Confirm source coupling tables, experimental conditions, and the homotaurine temperature label.
- Add and justify literature comparison methods; run them on the same saved splits with comparable tuning budgets and expression complexity measures.
- Report analytical, OLS, learned-network, and rounded-formula errors separately.
- Resolve the four-site verification baseline discrepancy documented in the reproduction guide before quoting reproduced accuracy.
- Recover the original architecture-search table and primary benchmark checkpoints if exact historical-run reproduction is required.
