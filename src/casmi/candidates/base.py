"""Candidate generation — what you can possibly answer, before ranking.

Candidate recall is the hard ceiling on MRR: a structure never proposed can
never be ranked. Keep this stage and the ranking stage separate so you always
know which one is costing you score.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

from ..features.base import MoleculeQuery


class CandidateSource:
    """Proposes structures for one molecule. Ordering is a weak prior only."""
    name: str = "unnamed"

    def fit(self, train_df) -> "CandidateSource":
        return self

    def propose(self, q: MoleculeQuery, limit: int = 1000) -> list[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


CANDIDATE_SOURCES: dict[str, Callable[..., CandidateSource]] = {}


def register_candidates(name: str):
    def deco(cls):
        CANDIDATE_SOURCES[name] = cls
        cls.name = name
        return cls
    return deco


def get_candidates(name: str, **kw) -> CandidateSource:
    if name not in CANDIDATE_SOURCES:
        raise KeyError(f"unknown candidate source {name!r}; have {sorted(CANDIDATE_SOURCES)}")
    return CANDIDATE_SOURCES[name](**kw)


class UnionCandidates(CandidateSource):
    """Pool several sources, preserving first-seen order and de-duplicating."""
    def __init__(self, sources: Sequence[CandidateSource], name: str | None = None):
        self.sources = list(sources)
        self.name = name or "+".join(s.name for s in self.sources)

    def fit(self, train_df):
        for s in self.sources:
            s.fit(train_df)
        return self

    def propose(self, q: MoleculeQuery, limit: int = 1000) -> list[str]:
        seen, out = set(), []
        for s in self.sources:
            for c in s.propose(q, limit):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
                    if len(out) >= limit:
                        return out
        return out


class BlockedCandidates(CandidateSource):
    """Wraps a source and hides blocked InChIKey14s.

    The validation splitter uses this to make class-3 molecules genuinely
    unreachable by retrieval. Without it, class-3 scores are fiction.
    """
    def __init__(self, inner: CandidateSource, blocked_keys: Iterable[str], key_of):
        self.inner, self.blocked, self.key_of = inner, frozenset(blocked_keys), key_of
        self.name = f"blocked({inner.name})"

    def fit(self, train_df):
        self.inner.fit(train_df)
        return self

    def propose(self, q: MoleculeQuery, limit: int = 1000) -> list[str]:
        if not self.blocked:
            return self.inner.propose(q, limit)
        out = [c for c in self.inner.propose(q, limit * 2) if self.key_of(c) not in self.blocked]
        return out[:limit]
