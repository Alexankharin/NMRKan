# Experiment configurations

`paper/` contains the supported manuscript protocols. Copy a JSON file to change its sample count, model seed, architecture, or training schedule; the effective configuration is saved with every run.

| File | Sampling | ASN and training |
|---|---|---|
| `paper/three_site.json` | 100,000 independent ratio pairs in [0,2]² | (2,16,4), 1000 epochs, MAE |
| `paper/three_site_verification.json` | 20,000 ratio pairs | (2,16,4), 400 epochs, MAE |
| `paper/four_site.json` | 10,000 physical coupling pairs; tracked/interpolated grid | (1,10,8), 5000 epochs, MSE |
| `paper/four_site_verification.json` | 10,000 uniform ratios in [0,3]; direct diagonalization | (1,10,8), 5000 epochs, MSE |

All use data and split seed 42. The new runner explicitly chooses model seed 42 and the final checkpoint. Exact original neural checkpoints were not supplied. Molecular training settings are controlled by `python -m experiments.molecules.train --help` and saved as `config.json` inside each domain.

`exploratory/` preserves earlier generic YAML configurations. These use different objectives/settings and must not be substituted for the manuscript protocols.
