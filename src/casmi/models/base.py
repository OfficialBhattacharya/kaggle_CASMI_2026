"""Model framework: a registry of named rankers.

Every model in this competition, however it works internally, has the same
shape — evidence for one molecule in, a ranked list of at most 25 SMILES out.
Pinning that contract is what lets a library search, a retrieval reranker and a
de-novo generator be compared on one axis, and combined by rank fusion.

Add a model:

    @register_model("my_model")
    class MyModel(Ranker):
        def fit(self, train_df, featurizer=None): ...
        def rank(self, q: MoleculeQuery, k: int = 25) -> list[str]: ...
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..config import MAX_GUESSES
from ..features.base import Featurizer, MoleculeQuery


class Ranker:
    """Base class for anything that answers a MoleculeQuery."""
    name: str = "unnamed"

    def fit(self, train_df, featurizer: Featurizer | None = None) -> "Ranker":
        return self

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        """Return at most k SMILES, best guess first."""
        raise NotImplementedError

    def score_candidates(self, q: MoleculeQuery, candidates: Sequence[str]) -> np.ndarray:
        """Optional: score a supplied candidate list. Enables reranking and fusion."""
        raise NotImplementedError(f"{type(self).__name__} cannot score arbitrary candidates")

    def rebind_sources(self, fn) -> None:
        """Apply `fn` to every CandidateSource this ranker owns, in place.

        `run_experiment` uses this to point sources at the full structure
        universe and apply class-3 blocking. Composite rankers must forward the
        call, or a nested source keeps a train-only database and silently scores
        zero on every class-2 molecule.
        """
        if getattr(self, "source", None) is not None:
            self.source = fn(self.source)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


MODELS: dict[str, Callable[..., Ranker]] = {}


def register_model(name: str):
    def deco(cls):
        MODELS[name] = cls
        cls.name = name
        return cls
    return deco


def get_model(name: str, **kw) -> Ranker:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODELS)}")
    return MODELS[name](**kw)


class RankFusion(Ranker):
    """Reciprocal-rank fusion of several rankers.

    The three novelty classes reward different machinery — library search wins
    class 1, retrieval wins class 2, generation is the only shot at class 3 —
    so the submission you actually want is almost certainly a fusion. RRF needs
    no score calibration between members, which is why it is the default.

    score(c) = sum_m  weight_m / (rrf_k + rank_m(c))
    """
    def __init__(self, members: Sequence[Ranker], weights: Sequence[float] | None = None,
                 rrf_k: int = 60, name: str | None = None, pool: int = 200):
        self.members = list(members)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.members)
        if len(self.weights) != len(self.members):
            raise ValueError("weights must match members")
        self.rrf_k, self.pool = rrf_k, pool
        self.name = name or "rrf(" + "+".join(m.name for m in self.members) + ")"

    def fit(self, train_df, featurizer=None):
        for m in self.members:
            m.fit(train_df, featurizer)
        return self

    def rebind_sources(self, fn) -> None:
        for m in self.members:
            m.rebind_sources(fn)

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        scores: dict[str, float] = {}
        for member, w in zip(self.members, self.weights):
            for rank, smi in enumerate(member.rank(q, self.pool), start=1):
                scores[smi] = scores.get(smi, 0.0) + w / (self.rrf_k + rank)
        return [s for s, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]


class PaddedRanker(Ranker):
    """Wraps a ranker and tops short lists up to k from a fallback.

    There is no penalty for a wrong guess beyond the slot it occupies, so
    returning fewer than 25 candidates is leaving free expected score behind.
    """
    def __init__(self, inner: Ranker, fallback: Ranker, name: str | None = None):
        self.inner, self.fallback = inner, fallback
        self.name = name or f"padded({inner.name})"

    def fit(self, train_df, featurizer=None):
        self.inner.fit(train_df, featurizer)
        self.fallback.fit(train_df, featurizer)
        return self

    def rebind_sources(self, fn) -> None:
        self.inner.rebind_sources(fn)
        self.fallback.rebind_sources(fn)

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        out = list(self.inner.rank(q, k))
        if len(out) >= k:
            return out[:k]
        seen = set(out)
        for smi in self.fallback.rank(q, k * 4):
            if smi not in seen:
                seen.add(smi)
                out.append(smi)
                if len(out) >= k:
                    break
        return out[:k]
