# Contributing

Keep reusable mathematics and model code in `nmrkan/`; put executable experiment orchestration in `experiments/`. Use `python -m ...` entry points and editable installation rather than adding working directories to `sys.path`. Retain existing public imports when a structural change does not require an API change.

Each paper experiment needs a named configuration, deterministic sampling and split seeds, explicit units and branch ordering, and a row in the reproduction guide. Save parameters, data/split provenance, metrics, and formulas in a new directory under `outputs/`. Keep full-precision neural metrics separate from rounded symbolic metrics. Compare regressors on matching splits.

Treat `results/reference/` and `data/raw/` as preserved inputs. New reference results should be deliberately promoted from a checked run with provenance and updated checksums. Do not overwrite historical artifacts or move generated checkpoints beside source modules. Old investigations belong in `archive/` with their limitations identified.

Run `python -m pytest` and `python -m experiments.check_notebooks` before committing changes to active workflows. Long training is separate from CI; a short training test verifies execution and extraction, not publication accuracy. Do not describe the archived exploratory scripts as supported paper reproduction commands.
