from __future__ import annotations

import sys
from pathlib import Path

import nbformat


def normalize_notebook(path: Path) -> None:
    nb = nbformat.read(path, as_version=4)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nbformat.write(nb, path)
    print(f"[OK] Normalized notebook: {path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("notebooks/03_mining_or_clustering.ipynb")
    normalize_notebook(target)

