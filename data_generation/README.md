# Synthetic data generation

`nmr_datagen.py` provides the original reduced three-site Hamiltonian and frequency helpers. `eigenvalues_4CH2.py` defines the eight-dimensional four-site block and eigenvector-overlap surface tracking. `datasets.py` supplies dataset builders, including the physical-grid interpolation used by the primary four-site experiment.

The paper runner uses these Hamiltonians explicitly and sorts dimensionless eigenvalues in descending order. Real-molecule Hamiltonians with unequal geminal couplings and independent projection checks live in `nmrkan/physics/`. The old frequency helper branch assignments should not be used as a substitute for the paper's ordered eigenvalue labels.
