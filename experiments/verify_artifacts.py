"""Verify that the preserved molecular reference files have not changed."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "results/reference/manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]
        ):
            failures.append(entry["path"])
    if failures:
        raise SystemExit("Missing or modified reference files:\n" + "\n".join(failures))
    print(f"Verified {len(manifest['files'])} reference artifacts.")


if __name__ == "__main__":
    main()
