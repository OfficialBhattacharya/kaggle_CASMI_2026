#!/usr/bin/env python3
"""Push notebooks (and optionally the framework) to Kaggle.

    python3 scripts/sync_kaggle.py --check              # verify auth + rules accepted
    python3 scripts/sync_kaggle.py --push eda           # push notebooks/01_eda.ipynb
    python3 scripts/sync_kaggle.py --push baseline
    python3 scripts/sync_kaggle.py --push-utils         # src/casmi as a private Dataset
    python3 scripts/sync_kaggle.py --status eda

Auth: a KGAT_ token at ~/.kaggle/access_token (chmod 600), from
kaggle.com -> Settings -> API -> Create New Token.

Talks to the REST API directly rather than shelling out to the `kaggle` CLI:
that package caps at 1.7.4.5 on Python 3.9, and 1.7.4.5 predates KGAT_ tokens
(it hard-requires a legacy kaggle.json and fails at import without one).

Notebooks are pushed **private** and with internet disabled, matching the code
competition's requirements. Make one public only when you mean to.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_api import (KaggleError, competition_files, kernel_status,  # noqa: E402
                        list_kernels, push_kernel, submissions, whoami)

ROOT = Path(__file__).resolve().parents[1]
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

NOTEBOOKS = {
    "eda": ("01_eda.ipynb", "CASMI26 EDA - data, task and metric", False),
    "baseline": ("02_baseline_submission.ipynb", "CASMI26 baseline submission", False),
}


def username() -> str:
    u = whoami()
    if not u:
        raise SystemExit(
            "Kaggle username unknown. Set it once:\n"
            "  echo <your-kaggle-username> > ~/.kaggle/username")
    return u


def check() -> int:
    print(f"username: {username()}")
    try:
        files = competition_files(COMP)
    except KaggleError as e:
        print(e)
        print(f"\nAccept the rules first: https://www.kaggle.com/competitions/{COMP}/rules")
        return 1
    print(f"data access OK — {len(files)} files:")
    for f in files:
        print(f"  {f['name']:28s} {f['totalBytes']/1e6:10.1f} MB")
    subs = submissions(COMP)
    print(f"\n{len(subs)} submission(s) so far:")
    for sb in subs[:10]:
        print(f"  {sb.get('date','')[:19]}  {str(sb.get('status')):10s} "
              f"public={sb.get('publicScore') or '-':>8}  {sb.get('description','')[:60]}")
    return 0


def push_notebook(which: str, public: bool = False) -> int:
    if which not in NOTEBOOKS:
        raise SystemExit(f"unknown notebook {which!r}; have {sorted(NOTEBOOKS)}")
    fname, title, gpu = NOTEBOOKS[which]
    src = ROOT / "notebooks" / fname
    if not src.exists():
        raise SystemExit(f"{src} missing — run scripts/build_{which}_notebook.py")

    slug = f"{username()}/casmi26-{which}"
    print(f"pushing {fname} -> {slug}  (private={not public}, internet=off)")
    try:
        # Internet stays off: a code competition rejects any notebook that has it on.
        r = push_kernel(slug=slug, title=title, notebook_path=src, competition=COMP,
                        private=not public, gpu=gpu, internet=False)
    except KaggleError as e:
        print(e)
        return 1
    print(f"  version : {r.get('versionNumber')}")
    if r.get("error"):
        print(f"  ERROR   : {r['error']}")
        return 1
    print(f"\nopen: {r.get('url') or 'https://www.kaggle.com/' + slug}")
    print("It queues and runs on Kaggle; check progress with --status.")
    return 0


def push_utils() -> int:
    """Publish src/casmi as a private Kaggle Dataset.

    Code competitions run offline, so this is how a notebook gets the framework:
    attach the dataset, then `sys.path.insert(0, '/kaggle/input/<slug>')`.

    Dataset upload needs multipart file handling the REST helper does not do, so
    this path still shells out to the CLI and therefore needs a legacy
    kaggle.json. Not required for either notebook.
    """
    import json, shutil, subprocess, tempfile
    who = username()
    slug = f"{who}/casmi26-utils"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        shutil.copytree(ROOT / "src" / "casmi", d / "casmi",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": "CASMI26 utils", "id": slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2))
        cli = ["kaggle"]
        r = subprocess.run(cli + ["datasets", "status", slug], capture_output=True, text=True)
        exists = r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()
        if exists:
            r = subprocess.run(cli + ["datasets", "version", "-p", str(d),
                                      "-m", "update framework", "-r", "zip"],
                               capture_output=True, text=True)
        else:
            r = subprocess.run(cli + ["datasets", "create", "-p", str(d), "-r", "zip"],
                               capture_output=True, text=True)
    print((r.stdout or "").strip() or (r.stderr or "").strip())
    return 0 if r.returncode == 0 else 1


def status(which: str) -> int:
    """Report run status, falling back to a listing when the token lacks scope.

    A KGAT_ token created without the `kernels.get` permission can push a
    notebook but cannot read its run status. The listing endpoint still shows
    the kernel exists, which is usually enough to confirm a push landed; for the
    actual run result, open the URL.
    """
    slug = f"{username()}/casmi26-{which}"
    try:
        r = kernel_status(slug)
        print(f"{slug}: {r.get('status')}")
        if r.get("failureMessage"):
            print(f"  failure: {r['failureMessage']}")
        return 0
    except KaggleError as e:
        if "kernels.get" not in str(e):
            print(e)
            return 1
        print(f"token lacks the 'kernels.get' scope — cannot read run status.")
        found = [k for k in list_kernels() if f"casmi26-{which}" in (k.get("ref") or "")]
        for k in found:
            print(f"  exists: {k['ref']}  (open it to see the run result)")
            print(f"          https://www.kaggle.com/code/{k['ref']}")
        if not found:
            print(f"  no kernel matching casmi26-{which} in the account listing")
        print("\n  To enable status here, create a token with kernel read permission.")
        return 0


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
