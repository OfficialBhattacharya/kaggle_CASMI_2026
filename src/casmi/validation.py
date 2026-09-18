"""Local validation that simulates the hidden test set.

The test set mixes three novelty classes, and a split that ignores them will
lie to you. A plain random spectrum split leaves other spectra of the same
molecule in train, so it measures class 1 only — and class 1 is the easy third.

`make_split` therefore builds a held-out set where each molecule is *assigned*
a simulated class and the training view is degraded accordingly:

  class 1  spectra of the molecule remain in train  (library search can win)
  class 2  every spectrum of the molecule is removed, structure still in the
           candidate database                        (retrieval + rerank)
  class 3  every spectrum removed AND the structure blocked from the candidate
           database                                  (de novo only)

Honouring `blocked_keys` is the candidate generator's responsibility; see
`Split.filter_candidates`. Skipping it inflates class-3 scores to nonsense.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import KEY_COL, NATURAL_PRODUCT_LIBS, TIMSTOF_LIBS


@dataclass
class Split:
    """One validation fold."""
    train_idx: np.ndarray            # positions into the source frame
    eval_idx: np.ndarray
    truth: dict[str, str]            # molecule_id -> true inchikey14
    novelty_class: dict[str, str]    # molecule_id -> "1" | "2" | "3"
    blocked_keys: frozenset[str] = frozenset()
    name: str = "fold0"

    def filter_candidates(self, candidates: Iterable[str], key_of) -> list[str]:
        """Drop class-3 blocked structures from a candidate list.

        Call this inside every candidate source during validation, or class-3
        numbers are meaningless. `key_of` maps a candidate to its InChIKey14.
        """
        if not self.blocked_keys:
            return list(candidates)
        return [c for c in candidates if key_of(c) not in self.blocked_keys]

    def summary(self) -> str:
        counts = pd.Series(self.novelty_class).value_counts().sort_index()
        return (f"{self.name}: {len(self.train_idx):,} train spectra / "
                f"{len(self.eval_idx):,} eval spectra, {len(self.truth)} molecules "
                f"[" + ", ".join(f"class {c}: {n}" for c, n in counts.items()) + "]")


def _molecule_table(df: pd.DataFrame, id_col: str, key_col: str) -> pd.DataFrame:
    """One row per molecule with the metadata the sampler weights on."""
    g = df.groupby(id_col, sort=False)
    out = pd.DataFrame({
        key_col: g[key_col].first(),
        "n_spectra": g.size(),
    })
    if "ingest_lib" in df.columns:
        out["libs"] = g["ingest_lib"].agg(lambda s: set(s.dropna()))
    if "precursor_mz" in df.columns:
        out["precursor_mz"] = g["precursor_mz"].median()
    return out.reset_index()


def make_split(
    df: pd.DataFrame,
    *,
    n_molecules: int = 400,
    class_ratio: Sequence[float] = (0.4, 0.4, 0.2),
    id_col: str = "molecule_id",
    key_col: str = KEY_COL,
    prefer_libs: Sequence[str] = NATURAL_PRODUCT_LIBS,
    prefer_weight: float = 4.0,
    max_spectra_per_molecule: int = 16,
    seed: int = 0,
    name: str = "fold0",
) -> Split:
    """Carve a test-set-shaped validation fold out of a labelled frame.

    Defaults mirror the stated test set: ~400 molecules, 1-16 spectra each.
    `class_ratio` is a guess — the real mix is hidden for the whole competition,
    so always read the per-class numbers, not just the headline MRR.

    Molecules from `prefer_libs` are up-weighted because the test set is natural
    products on a timsTOF, while most training spectra are synthetic screening
    compounds. Sampling uniformly would validate on the wrong chemistry.
    """
    if abs(sum(class_ratio) - 1.0) > 1e-6:
        raise ValueError(f"class_ratio must sum to 1, got {sum(class_ratio)}")
    if key_col not in df.columns:
        raise KeyError(f"{key_col!r} missing — make_split needs labelled data")

    rng = np.random.default_rng(seed)
    mols = _molecule_table(df, id_col, key_col)
    mols = mols[mols[key_col].notna()]

    # Only molecules with >=2 spectra can be class 1 (one spectrum held out,
    # at least one left behind for the library to match against).
    weights = np.ones(len(mols))
    if "libs" in mols.columns and prefer_weight != 1.0:
        pref = set(prefer_libs)
        weights = np.where(mols["libs"].map(lambda s: bool(s & pref)), prefer_weight, 1.0)
    weights = weights / weights.sum()

    n_pick = min(n_molecules, len(mols))
    picked = rng.choice(len(mols), size=n_pick, replace=False, p=weights)
    chosen = mols.iloc[picked].reset_index(drop=True)

    # Assign classes; class 1 needs a spare spectrum, so reassign where impossible.
    classes = rng.choice(["1", "2", "3"], size=n_pick, p=list(class_ratio))
    classes = np.where((classes == "1") & (chosen["n_spectra"].to_numpy() < 2), "2", classes)
    chosen["cls"] = classes

    eval_positions: list[np.ndarray] = []
    drop_positions: list[np.ndarray] = []
    pos_by_mol = {m: np.asarray(p) for m, p in
                  df.reset_index(drop=True).groupby(id_col, sort=False).indices.items()}

    for mol_id, cls, in zip(chosen[id_col], chosen["cls"]):
        pos = pos_by_mol[mol_id]
        n_eval = int(min(len(pos), rng.integers(1, max_spectra_per_molecule + 1)))
        if cls == "1":
            n_eval = min(n_eval, len(pos) - 1)   # leave >=1 behind in train
            n_eval = max(n_eval, 1)
        taken = rng.choice(pos, size=n_eval, replace=False)
        eval_positions.append(taken)
        # classes 2 and 3: every spectrum of this molecule leaves the train view
        drop_positions.append(pos if cls in ("2", "3") else taken)

    eval_idx = np.sort(np.concatenate(eval_positions))
    dropped = np.concatenate(drop_positions)
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[dropped] = False

    blocked = frozenset(chosen.loc[chosen["cls"] == "3", key_col])
    return Split(
        train_idx=np.flatnonzero(train_mask),
        eval_idx=eval_idx,
        truth=dict(zip(chosen[id_col], chosen[key_col])),
        novelty_class=dict(zip(chosen[id_col], chosen["cls"])),
        blocked_keys=blocked,
        name=name,
    )


def make_splits(df: pd.DataFrame, n_folds: int = 3, seed: int = 0, **kw) -> list[Split]:
    """Independent folds. Spectra overlap between folds — these are repeated
    resamples for variance estimation, not a partition of the data."""
    return [make_split(df, seed=seed + i, name=f"fold{i}", **kw) for i in range(n_folds)]


def timstof_holdout(df: pd.DataFrame, **kw) -> Split:
    """Validate only on the two timsTOF libraries, the closest match to test."""
    if "ingest_lib" not in df.columns:
        raise KeyError("ingest_lib missing")
    kw.setdefault("prefer_libs", TIMSTOF_LIBS)
    kw.setdefault("prefer_weight", 50.0)
    kw.setdefault("name", "timstof")
    return make_split(df, **kw)
