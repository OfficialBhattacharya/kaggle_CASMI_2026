#!/usr/bin/env python3
"""Run a feature or model experiment from the command line.

    python3 scripts/run_experiment.py --list
    python3 scripts/run_experiment.py --model library_search --candidates mass_window
    python3 scripts/run_experiment.py --features binned,neutral_loss,meta
    python3 scripts/run_experiment.py --leaderboard
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from casmi.candidates import CANDIDATE_SOURCES, get_candidates
from casmi.config import PATHS
from casmi.experiment import leaderboard, run_feature_experiment, run_model_experiment
from casmi.features import FEATURIZERS, FeatureUnion, get_featurizer
from casmi.models import MODELS, get_model
from casmi.spectra import CleanConfig
from casmi.validation import make_split


def load_train(limit: int | None) -> pd.DataFrame:
    path = PATHS.train
    if not path.exists():
        raise SystemExit(
            f"train.parquet not found at {path}\n"
            f"  set CASMI_DATA, or download:\n"
            f"  kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra "
            f"-p data --unzip")
    df = pd.read_parquet(path)
    if limit and len(df) > limit:
        # sample whole molecules, never individual spectra — splitting a molecule
        # across the sample boundary would fabricate class-1 evidence
        mols = df.molecule_id.drop_duplicates()
        keep = mols.sample(frac=min(1.0, limit / len(df)), random_state=0)
        df = df[df.molecule_id.isin(set(keep))]
    return df.reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="show registered components")
    p.add_argument("--leaderboard", action="store_true", help="show local results so far")
    p.add_argument("--model", help="model name from the registry")
    p.add_argument("--features", help="comma-separated featurizer names")
    p.add_argument("--candidates", default=None, help="candidate source name")
    p.add_argument("--limit", type=int, default=400_000, help="max spectra to load (0 = all)")
    p.add_argument("--molecules", type=int, default=400, help="held-out molecules")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bin-width", type=float, default=1.0)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    if args.list:
        print("featurizers      :", ", ".join(sorted(FEATURIZERS)))
        print("models           :", ", ".join(sorted(MODELS)))
        print("candidate sources:", ", ".join(sorted(CANDIDATE_SOURCES)))
        return 0

    if args.leaderboard:
        lb = leaderboard()
        if lb.empty:
            print("no experiments logged yet")
            return 0
        cols = [c for c in ["name", "kind", "mrr", "mrr_class1", "mrr_class2", "mrr_class3",
                            "candidate_recall", "n", "seconds", "run_at"] if c in lb.columns]
        print(lb[cols].to_string(index=False))
        return 0

    if not args.model and not args.features:
        p.error("give --model or --features (or --list / --leaderboard)")

    df = load_train(args.limit or None)
    print(f"loaded {len(df):,} spectra / {df.molecule_id.nunique():,} molecules")
    split = make_split(df, n_molecules=args.molecules, seed=args.seed)
    print(split.summary())

    clean = CleanConfig()
    if args.features:
        names = [n.strip() for n in args.features.split(",") if n.strip()]
        parts = [get_featurizer(n, bin_width=args.bin_width)
                 if n in ("binned", "neutral_loss") else get_featurizer(n) for n in names]
        f = parts[0] if len(parts) == 1 else FeatureUnion(parts)
        run_feature_experiment(df, f, name=args.name, split=split, clean=clean)
    else:
        cands = get_candidates(args.candidates) if args.candidates else None
        run_model_experiment(df, get_model(args.model), name=args.name, split=split,
                             clean=clean, candidates=cands)
    print(f"\nlogged to {PATHS.results / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
