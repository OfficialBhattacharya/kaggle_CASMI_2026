#!/usr/bin/env python3
"""Validate a submission.csv against every rule the grader enforces.

    python3 scripts/validate_submission.py submission.csv [--test data/test.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from casmi.config import PATHS
from casmi.submit import SubmissionError, check_smiles_parse, validate_submission


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--test", default=None, help="test.parquet, to check molecule_id coverage")
    args = p.parse_args()

    df = pd.read_csv(args.path)
    print(f"{args.path}: {len(df)} rows, columns {list(df.columns)}")

    expected = None
    test_path = Path(args.test) if args.test else PATHS.test
    if test_path.exists():
        expected = pd.read_parquet(test_path, columns=["molecule_id"]).molecule_id.unique()
        print(f"checking against {len(expected)} molecule_ids in {test_path}")

    try:
        validate_submission(df, expected)
    except SubmissionError as e:
        print(f"\nREJECTED: {e}")
        return 1

    n = df.smiles.astype(str).str.count(";") + 1
    print(f"\nVALID")
    print(f"  guesses per row: min={n.min()} median={n.median():.0f} max={n.max()}")
    if (n < 25).any():
        print(f"  {(n < 25).sum()} row(s) use fewer than 25 slots — unused slots are free score")
    parse = check_smiles_parse(df)
    if parse.get("rdkit"):
        print(f"  RDKit parse check: {parse['unparseable']}/{parse['checked']} unparseable "
              f"({parse['frac_bad']:.2%})")
    else:
        print("  RDKit not installed — skipped SMILES parse check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
