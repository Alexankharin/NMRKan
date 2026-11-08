"""Execute only the supported, numbered notebooks in isolated Python kernels."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/executed_notebooks"
    output.mkdir(parents=True, exist_ok=True)
    # Keep runtime files inside the checkout rather than the user's home.
    for key, name in (
        ("JUPYTER_RUNTIME_DIR", "runtime"),
        ("IPYTHONDIR", "ipython"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        folder = output / name
        folder.mkdir(exist_ok=True)
        os.environ.setdefault(key, str(folder))
    os.environ.setdefault("MPLBACKEND", "Agg")
    for path in sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        print(f"Executing {path.name}", flush=True)
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbformat.write(notebook, output / path.name)
    print(f"Executed notebooks: {output}")


if __name__ == "__main__":
    main()
