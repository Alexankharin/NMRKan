# Reusable Python library

- `models.py`, `activations.py`: ASN layers, sparse activation selection and algebraic formula extraction. `KharKAN` is retained as the public class name.
- `physics/three_ch2.py`: molecular even/odd Hamiltonians, normalization, supplied coupling rows and independent six-spin checks.
- `physics/real_molecules.py`: exact analytical solution, perturbative comparisons, observables and synthetic spectra.
- `training/`, `config/`, `data/`: earlier general-purpose trainers and configuration/data helpers.
- `analysis/`, `dimensional_analysis.py`, `sympy_torch.py`: symbolic extraction and dimensional analysis.
- `utils/`: plotting, metrics, checkpoint helpers and analytical utilities.

Experiment-specific orchestration belongs in `experiments/`. The supported paper and molecular workflows use explicit eigenvalue conventions described in `docs/reproducing_the_paper.md`; generic historical frequency helpers are not interchangeable with those eigenvalue targets.
