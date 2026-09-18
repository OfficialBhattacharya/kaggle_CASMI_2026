#!/usr/bin/env python3
"""Strip outputs and execution counts from notebooks before committing.

A committed .ipynb full of output blobs is unreviewable and makes every diff a
merge conflict — which matters here because the point is to work from several
devices. Run before `git add`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
changed = []
for path in sorted((root / "notebooks").glob("*.ipynb")):
    nb = nbformat.read(str(path), as_version=4)
    dirty = False
    for cell in nb.cells:
        if cell.cell_type == "code":
            if cell.get("outputs"):
                cell["outputs"] = []; dirty = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None; dirty = True
    nb.metadata.pop("widgets", None)
    if dirty:
        nbformat.write(nb, str(path))
        changed.append(path.name)
    print(f"{'stripped' if dirty else 'clean   '}  {path.name}")

sys.exit(0)
