# Historical investigations

These files preserve earlier work and are not the supported reproduction path. Start with the numbered notebooks and `docs/reproducing_the_paper.md` instead.

- `notebooks/`: original exploratory notebooks with their saved outputs; some depend on removed modules or external results. `EDA_2025_regions.ipynb.fragment.txt` was already malformed notebook JSON and is retained verbatim as a text fragment.
- `scripts/`: earlier grid searches, genetic search and multi-experiment runners.
- `docs/`: previous migration, checkpointing and normalization notes. References to `main.py` or `experiments/run_experiments.py` are historical; those entry points were absent before cleanup.
- `logs/`: supplied training, validation and report-build logs.
- `relocation.json`: old/new paths and SHA-256 values of files before reorganization. Source files migrated into active packages subsequently received import and path updates; raw data and saved reference results did not.

Archive content is excluded from notebook execution and package discovery. The original raw data is now in `data/raw/2024/` and `data/raw/2025/`; the original `ANSWER/` contents are split between `paper/`, `nmrkan/physics/`, `experiments/molecules/`, `tests/molecules/`, and `results/reference/molecules/`.
