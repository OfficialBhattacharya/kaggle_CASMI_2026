"""MRR@25 — the competition metric, plus per-slice breakdowns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import MAX_GUESSES


def reciprocal_rank(ranked_keys: Sequence[str | None], true_key: str, k: int = MAX_GUESSES) -> float:
    """1/rank of the first correct key in the first k guesses, else 0.

    Keys are InChIKey14s. Duplicates are NOT collapsed: a repeated guess wastes
    a slot on the real leaderboard, so it must waste one here too.
    """
    for i, key in enumerate(ranked_keys[:k], start=1):
        if key is not None and key == true_key:
            return 1.0 / i
    return 0.0


def mrr_at_k(
    predictions: Mapping[str, Sequence[str | None]],
    truth: Mapping[str, str],
    k: int = MAX_GUESSES,
) -> float:
    """Mean reciprocal rank over every molecule in `truth`.

    A molecule missing from `predictions` scores 0 rather than being skipped —
    on the leaderboard an absent molecule_id is a rejected submission, and
    silently dropping it here would flatter the score.
    """
    if not truth:
        return 0.0
    return float(np.mean([
        reciprocal_rank(predictions.get(mid, ()), key, k) for mid, key in truth.items()
    ]))


@dataclass
class ScoreReport:
    """MRR@25 overall, sliced by novelty class, plus diagnostics."""
    mrr: float
    n: int
    by_class: dict[str, float] = field(default_factory=dict)
    n_by_class: dict[str, int] = field(default_factory=dict)
    hit_rate_at: dict[int, float] = field(default_factory=dict)
    candidate_recall: float = float("nan")   # truth present anywhere in candidates
    mean_candidates: float = float("nan")

    def to_row(self) -> dict:
        row = {"mrr": self.mrr, "n": self.n,
               "candidate_recall": self.candidate_recall,
               "mean_candidates": self.mean_candidates}
        row.update({f"mrr_class{c}": v for c, v in sorted(self.by_class.items())})
        row.update({f"n_class{c}": v for c, v in sorted(self.n_by_class.items())})
        row.update({f"hit@{k}": v for k, v in sorted(self.hit_rate_at.items())})
        return row

    def __str__(self) -> str:
        parts = [f"MRR@25 = {self.mrr:.4f}  (n={self.n})"]
        for c in sorted(self.by_class):
            parts.append(f"  class {c}: {self.by_class[c]:.4f}  (n={self.n_by_class.get(c, 0)})")
        if self.hit_rate_at:
            parts.append("  hit-rate " + "  ".join(
                f"@{k}={v:.3f}" for k, v in sorted(self.hit_rate_at.items())))
        if not np.isnan(self.candidate_recall):
            parts.append(f"  candidate recall = {self.candidate_recall:.4f} "
                         f"(ceiling on MRR), mean candidates = {self.mean_candidates:.0f}")
        return "\n".join(parts)


def score(
    predictions: Mapping[str, Sequence[str | None]],
    truth: Mapping[str, str],
    novelty_class: Mapping[str, str] | None = None,
    candidate_pools: Mapping[str, set] | None = None,
    hit_ks: Sequence[int] = (1, 5, 25),
) -> ScoreReport:
    """Full evaluation.

    `candidate_pools` (molecule -> set of InChIKey14 offered by the candidate
    generator before ranking) is optional but worth passing: candidate recall is
    the hard ceiling on MRR, and separating 'never proposed' from 'proposed but
    ranked badly' is the single most useful diagnostic in this competition.
    """
    rrs = {mid: reciprocal_rank(predictions.get(mid, ()), key) for mid, key in truth.items()}
    rep = ScoreReport(mrr=float(np.mean(list(rrs.values()))) if rrs else 0.0, n=len(truth))

    for k in hit_ks:
        rep.hit_rate_at[k] = float(np.mean([
            reciprocal_rank(predictions.get(mid, ()), key, k) > 0 for mid, key in truth.items()
        ])) if truth else 0.0

    if novelty_class:
        df = pd.DataFrame({"rr": pd.Series(rrs),
                           "cls": pd.Series({m: novelty_class.get(m, "?") for m in rrs})})
        rep.by_class = df.groupby("cls")["rr"].mean().to_dict()
        rep.n_by_class = df.groupby("cls")["rr"].size().to_dict()

    if candidate_pools is not None and truth:
        rep.candidate_recall = float(np.mean([
            key in candidate_pools.get(mid, ()) for mid, key in truth.items()]))
        rep.mean_candidates = float(np.mean([len(candidate_pools.get(mid, ())) for mid in truth]))
    return rep
