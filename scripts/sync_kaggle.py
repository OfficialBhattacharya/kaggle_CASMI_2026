#!/usr/bin/env python3
"""Push notebooks (and optionally the framework) to Kaggle.

    python3 scripts/sync_kaggle.py --check              # verify auth + rules accepted
    python3 scripts/sync_kaggle.py --push eda           # push notebooks/01_eda.ipynb
    python3 scripts/sync_kaggle.py --push baseline
    python3 scripts/sync_kaggle.py --push-utils         # src/casmi as a private Dataset
    python3 scripts/sync_kaggle.py --status eda

Auth: put your Kaggle API token at ~/.kaggle/kaggle.json (chmod 600), from
kaggle.com -> Settings -> API -> Create New Token.

Notebooks are pushed **private** and with internet disabled, matching the code
competition's requirements. Make one public only when you mean to.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

NOTEBOOKS = {
    "eda": ("01_eda.ipynb", "CASMI26 EDA - data, task and metric", False),
    "baseline": ("02_baseline_submission.ipynb", "CASMI26 baseline submission", False),
}


def kaggle_cli() -> list[str]:
    """Locate the kaggle CLI, falling back to `python -m kaggle`."""
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    for cand in (Path.home() / "Library/Python/3.9/bin/kaggle",
                 Path.home() / ".local/bin/kaggle"):
        if cand.exists():
            return [str(cand)]
    return [sys.executable, "-m", "kaggle"]


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(args))
    return subprocess.run(args, capture_output=True, text=True, **kw)


def username() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return json.loads(cfg.read_text())["username"]
    raise SystemExit(
        "No Kaggle credentials found.\n"
        "  1. kaggle.com -> your avatar -> Settings -> API -> Create New Token\n"
        "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
        "  3. chmod 600 ~/.kaggle/kaggle.json")


def check() -> int:
    cli = kaggle_cli()
    print(f"kaggle CLI: {' '.join(cli)}")
    who = username()
    print(f"username  : {who}")
    r = run(cli + ["competitions", "files", "-c", COMP])
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[:800])
    if "403" in out or "rules" in out.lower():
        print("\n>>> You must accept the competition rules first (and click Join Competition):")
        print(f"    https://www.kaggle.com/competitions/{COMP}/rules")
        return 1
    print("\nauth OK and rules appear accepted")
    return 0


def push_notebook(which: str, public: bool = False) -> int:
    if which not in NOTEBOOKS:
        raise SystemExit(f"unknown notebook {which!r}; have {sorted(NOTEBOOKS)}")
    fname, title, gpu = NOTEBOOKS[which]
    src = ROOT / "notebooks" / fname
    if not src.exists():
        raise SystemExit(f"{src} missing — run scripts/build_{which}_notebook.py")

    who = username()
    slug = f"{who}/casmi26-{which}"
    meta = {
        "id": slug,
        "title": title,
        "code_file": fname,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": not public,
        "enable_gpu": gpu,
        "enable_tpu": False,
        # Code competition requirement: submissions must run with internet off.
        "enable_internet": False,
        "competition_sources": [COMP],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copy(src, d / fname)
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"pushing {fname} -> kaggle.com/code/{slug}  (private={not public})")
        r = run(kaggle_cli() + ["kernels", "push", "-p", str(d)])
    print((r.stdout or "").strip())
    if r.returncode != 0 or "error" in (r.stderr or "").lower():
        print((r.stderr or "").strip())
        return 1
    print(f"\nopen: https://www.kaggle.com/{slug}")
    print("It queues and runs on Kaggle; check progress with --status.")
    return 0


def push_utils() -> int:
    """Publish src/casmi as a private Kaggle Dataset.

    Code competitions run offline, so this is how a notebook gets the framework:
    attach the dataset, then `sys.path.insert(0, '/kaggle/input/<slug>')`.
    """
    who = username()
    slug = f"{who}/casmi26-utils"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copytree(ROOT / "src" / "casmi", d / "casmi",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": "CASMI26 utils", "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        cli = kaggle_cli()
        r = run(cli + ["datasets", "status", slug])
        exists = r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()
        if exists:
            r = run(cli + ["datasets", "version", "-p", str(d), "-m", "update framework", "-r", "zip"])
        else:
            r = run(cli + ["datasets", "create", "-p", str(d), "-r", "zip"])
    print((r.stdout or "").strip() or (r.stderr or "").strip())
    return 0 if r.returncode == 0 else 1


def status(which: str) -> int:
    who = username()
    r = run(kaggle_cli() + ["kernels", "status", f"{who}/casmi26-{which}"])
    print((r.stdout or "").strip() or (r.stderr or "").strip())
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--push", choices=sorted(NOTEBOOKS))
    p.add_argument("--push-utils", action="store_true")
    p.add_argument("--status", choices=sorted(NOTEBOOKS))
    p.add_argument("--public", action="store_true",
                   help="publish the notebook publicly (default: private)")
    a = p.parse_args()

    if a.check:      return check()
    if a.push:       return push_notebook(a.push, a.public)
    if a.push_utils: return push_utils()
    if a.status:     return status(a.status)
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
