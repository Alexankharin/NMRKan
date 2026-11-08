# Cleanup validation records

`paper_baselines.json` records the effective configurations and measured numerical results from the reconstructed paper workflows. `environment.json` records the local validation environment. These are new audit calculations, distinct from the supplied historical artifacts under `../reference/`.

The four-site uniform-ratio baseline does not reproduce the manuscript's quoted 0.00702 error. See `docs/reproducing_the_paper.md` and `docs/validation.md` for the observed discrepancy and reproduction commands. Neural paper training was checked only with short runs, not repeated for the full 1000/5000-epoch protocols during this cleanup.
