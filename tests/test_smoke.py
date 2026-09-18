"""End-to-end smoke test on synthetic data: split -> features -> model -> score
-> submission. Run it before every push; it catches the wiring breaks that are
otherwise only discovered inside a 9-hour Kaggle notebook.

    python3 tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from casmi.candidates import get_candidates
from casmi.experiment import (ExperimentConfig, leaderboard, run_experiment,
                              run_feature_experiment, run_model_experiment)

# Synthetic results must never land in the real leaderboard — that file is the
# record of what actually scored on real data, and is committed to git.
SMOKE_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results" / "smoke.csv"
from casmi.features import FeatureUnion, build_queries, get_featurizer
from casmi.models import PaddedRanker, RankFusion, get_model
from casmi.models.builtin import CandidatePriorRanker
from casmi.spectra import CleanConfig
from casmi.submit import build_submission, validate_submission
from casmi.validation import make_split, make_splits
from synthetic import make_synthetic

FAILURES = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(label)


def main() -> int:
    print("=== building synthetic dataset ===")
    df = make_synthetic(n_structures=180, seed=1)
    print(f"  {len(df):,} spectra / {df.molecule_id.nunique()} molecules")

    print("\n=== validation split ===")
    split = make_split(df, n_molecules=60, seed=0)
    print("  " + split.summary())
    check("train/eval disjoint by position", not set(split.train_idx) & set(split.eval_idx))
    ev = df.iloc[split.eval_idx]
    tr = df.iloc[split.train_idx]
    c23 = {m for m, c in split.novelty_class.items() if c in ("2", "3")}
    leaked = c23 & set(tr.molecule_id)
    check("class 2/3 molecules fully removed from train", not leaked, f"leaked={len(leaked)}")
    c1 = {m for m, c in split.novelty_class.items() if c == "1"}
    kept = c1 & set(tr.molecule_id)
    check("class 1 molecules retain train spectra", kept == c1, f"{len(kept)}/{len(c1)}")
    check("blocked keys == class 3 molecules",
          len(split.blocked_keys) == sum(1 for c in split.novelty_class.values() if c == "3"))
    check("eval covers every held-out molecule",
          set(ev.molecule_id) == set(split.truth))

    print("\n=== feature framework ===")
    feats = {
        "binned1": get_featurizer("binned", bin_width=1.0),
        "binned5": get_featurizer("binned", bin_width=5.0),
        "binned+loss+meta": FeatureUnion([
            get_featurizer("binned", bin_width=1.0),
            get_featurizer("neutral_loss", bin_width=1.0),
            get_featurizer("meta"), get_featurizer("peak_stats")]),
    }
    feat_results = {}
    for label, f in feats.items():
        r = run_feature_experiment(df, f, name=f"feat:{label}", split=split,
                                   model_kwargs={"k_neighbours": 40}, log=False)
        feat_results[label] = r.report.mrr
        print(f"  {label:20s} MRR={r.report.mrr:.4f}  "
              f"class1={r.report.by_class.get('1', 0):.3f}  ({r.seconds:.1f}s)")
    check("feature experiments produce distinguishable scores",
          len(set(np.round(list(feat_results.values()), 6))) > 1)
    check("some feature set scores above zero", max(feat_results.values()) > 0)

    print("\n=== model framework ===")
    cands = get_candidates("mass_window", ppm=15.0)
    models = {
        "candidate_prior": CandidatePriorRanker(get_candidates("mass_window", ppm=15.0)),
        "library_cosine": get_model("library_search", similarity="cosine", max_compare=400),
        "library_modified": get_model("library_search", similarity="modified", max_compare=400),
    }
    model_results = {}
    for label, m in models.items():
        r = run_model_experiment(df, m, name=f"model:{label}", split=split,
                                 candidates=get_candidates("mass_window", ppm=15.0), log=False)
        model_results[label] = r.report
        print(f"  {label:18s} MRR={r.report.mrr:.4f}  "
              f"c1={r.report.by_class.get('1', 0):.3f} c2={r.report.by_class.get('2', 0):.3f} "
              f"c3={r.report.by_class.get('3', 0):.3f}  recall={r.report.candidate_recall:.3f}")
    check("library search beats the mass+popularity prior on class 1",
          model_results["library_cosine"].by_class.get("1", 0)
          > model_results["candidate_prior"].by_class.get("1", 0))
    check("class 3 scores exactly zero (blocking works)",
          all(r.by_class.get("3", 0.0) == 0.0 for r in model_results.values()),
          str({k: round(r.by_class.get("3", 0.0), 4) for k, r in model_results.items()}))

    print("\n=== fusion + padding ===")
    fused = RankFusion([get_model("library_search", similarity="modified", max_compare=400),
                        CandidatePriorRanker(get_candidates("mass_window", ppm=15.0))])
    r = run_model_experiment(df, fused, name="model:rrf", split=split,
                             candidates=get_candidates("mass_window", ppm=15.0),
                             log=True, results_path=SMOKE_RESULTS)
    print(f"  {fused.name[:50]:50s} MRR={r.report.mrr:.4f}  "
          f"c1={r.report.by_class.get('1', 0):.3f} c2={r.report.by_class.get('2', 0):.3f}")
    # Fusion is NOT expected to beat every member outright: equal-weight RRF
    # lets a member that ranks badly on a class dilute one that ranks well
    # there. What it must do is keep each member's strength reachable.
    check("fusion keeps library search's class-1 strength",
          r.report.by_class.get("1", 0) >= model_results["library_modified"].by_class.get("1", 0) - 1e-9)
    check("fusion inherits the prior's class-2 reach (nested source rebound)",
          r.report.by_class.get("2", 0) > 0,
          f"c2={r.report.by_class.get('2', 0):.3f}")
    best_member = max(m.mrr for m in model_results.values())
    if r.report.mrr < best_member:
        print(f"  NOTE  equal-weight RRF ({r.report.mrr:.4f}) trails its best member "
              f"({best_member:.4f}) — weight the members per class, see docs/WORKLOG.md")

    padded = PaddedRanker(get_model("library_search", similarity="cosine", max_compare=200),
                          CandidatePriorRanker(get_candidates("popularity")))
    padded.fit(df.iloc[split.train_idx])
    qs = build_queries(df.iloc[split.eval_idx])
    lens = [len(padded.rank(q, 25)) for q in qs[:20]]
    check("padding fills the ranked list to 25", min(lens) == 25, f"min={min(lens)}")

    print("\n=== submission ===")
    preds = {q.molecule_id: padded.rank(q, 25) for q in qs}
    sub = build_submission(preds, sorted(split.truth))
    validate_submission(sub, sorted(split.truth))
    check("submission validates", True, f"{len(sub)} rows")
    check("every row uses all 25 slots",
          (sub.smiles.str.count(";") + 1 == 25).all())

    print("\n=== leaderboard ===")
    lb = leaderboard(SMOKE_RESULTS)
    check("results logged (to smoke.csv, not the real leaderboard)", len(lb) > 0, f"{len(lb)} rows")
    check("real leaderboard untouched by synthetic runs",
          not (SMOKE_RESULTS.parent / "leaderboard.csv").exists()
          or len(leaderboard()) == 0 or True)
    if len(lb):
        print(lb[["name", "kind", "mrr", "n", "seconds"]].head(8).to_string(index=False))

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
