"""Built-in candidate sources: mass-window retrieval over a structure database."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..chem import exact_mass, inchikey14, neutral_mass_from_precursor
from ..chem import RDKIT as RDKIT_AVAILABLE
from ..config import KEY_COL, LABEL_COL, NATURAL_PRODUCT_LIBS
from ..features.base import MoleculeQuery
from .base import CandidateSource, register_candidates


@register_candidates("mass_window")
class MassWindowCandidates(CandidateSource):
    """Every known structure whose neutral mass matches the query, by ppm window.

    Built from the training labels by default. Swap in a bigger structure table
    (PubChem / COCONUT subset shipped as a Kaggle dataset) via `structures=` to
    reach class-2 molecules that have no training spectra.

    `popularity` — how many training spectra back a structure — is the tie-break
    and doubles as a natural-product prior when built from NP libraries.
    """
    def __init__(self, ppm: float = 10.0, structures: pd.DataFrame | None = None,
                 prefer_libs=NATURAL_PRODUCT_LIBS, popularity_weight: float = 1.0):
        self.ppm, self.popularity_weight = ppm, popularity_weight
        self.prefer_libs = set(prefer_libs or ())
        self._structures = structures
        self.masses = np.empty(0)
        self.smiles: list[str] = []
        self.keys: list[str] = []
        self.pop = np.empty(0)
        self.name = f"mass_window(ppm={ppm})"

    def fit(self, train_df: pd.DataFrame | None = None):
        """Index structures by neutral monoisotopic mass.

        Mass comes from the `exact_mass` column when present (precomputed once
        on Kaggle where RDKit lives), else RDKit, else the labelled spectra's
        own precursor-implied mass — so this still works without RDKit.
        """
        src = self._structures
        if src is None:
            if train_df is None:
                raise ValueError("MassWindowCandidates needs train_df or structures=")
            src = self._structures_from_train(train_df)

        src = src[src[LABEL_COL].notna()].copy()
        if "exact_mass" not in src.columns:
            src["exact_mass"] = [exact_mass(s) for s in src[LABEL_COL]]
        elif RDKIT_AVAILABLE:   # prefer the true structural mass over the measured one
            rd = np.array([exact_mass(s) for s in src[LABEL_COL]])
            src["exact_mass"] = np.where(np.isfinite(rd), rd, src["exact_mass"])
        src = src[np.isfinite(src["exact_mass"])]
        if KEY_COL not in src.columns:
            src[KEY_COL] = [inchikey14(s) for s in src[LABEL_COL]]
        if "popularity" not in src.columns:
            src["popularity"] = 1.0

        src = src.sort_values("exact_mass").reset_index(drop=True)
        self.masses = src["exact_mass"].to_numpy(dtype=float)
        self.smiles = src[LABEL_COL].tolist()
        self.keys = src[KEY_COL].tolist()
        self.pop = src["popularity"].to_numpy(dtype=float)
        return self

    def _structures_from_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse the spectra frame to unique structures with a popularity count."""
        cols = {LABEL_COL: "first"}
        if KEY_COL in train_df.columns:
            group_key = KEY_COL
        else:
            group_key = LABEL_COL
            cols = {}
        g = train_df.groupby(group_key, sort=False)
        out = pd.DataFrame({LABEL_COL: g[LABEL_COL].first(), "popularity": g.size()})
        if "ingest_lib" in train_df.columns and self.prefer_libs:
            np_hits = g["ingest_lib"].agg(lambda s: len(set(s.dropna()) & self.prefer_libs))
            # a structure seen in natural-product libraries is a better prior for
            # this test set than one seen only in synthetic screening libraries
            out["popularity"] = out["popularity"] * (1.0 + np_hits)
        # Precursor-implied neutral mass, used when RDKit is unavailable to
        # compute an exact mass from the SMILES. Derived here rather than
        # required as a column so the source works on any labelled frame.
        if "neutral_mass" not in train_df.columns:
            train_df = train_df.assign(neutral_mass=[
                neutral_mass_from_precursor(p, a)
                for p, a in zip(train_df["precursor_mz"], train_df["adduct"])])
            g = train_df.groupby(group_key, sort=False)
        out["exact_mass"] = g["neutral_mass"].median()
        out = out.reset_index()
        if group_key == KEY_COL:
            out = out.rename(columns={KEY_COL: KEY_COL})
        return out

    def propose(self, q: MoleculeQuery, limit: int = 1000) -> list[str]:
        mass = q.consensus_mass()
        if not np.isfinite(mass) or self.masses.size == 0:
            return []
        tol = mass * self.ppm * 1e-6
        lo, hi = np.searchsorted(self.masses, [mass - tol, mass + tol])
        if hi <= lo:
            return []
        idx = np.arange(lo, hi)
        if self.popularity_weight:
            idx = idx[np.argsort(-self.pop[idx] * self.popularity_weight)]
        return [self.smiles[i] for i in idx[:limit]]

    def keys_for(self, smiles_list):
        """InChIKey14 for indexed structures without re-running RDKit."""
        from ..chem import KEY_RESOLVER
        lookup = dict(zip(self.smiles, self.keys))
        return [lookup.get(s) or KEY_RESOLVER(s) for s in smiles_list]


@register_candidates("popularity")
class PopularityCandidates(CandidateSource):
    """Globally most common structures, ignoring the spectrum entirely.

    A floor, not a method — but it is the right thing to pad a short ranked list
    with, since an unused slot scores exactly nothing.
    """
    def __init__(self, prefer_libs=NATURAL_PRODUCT_LIBS):
        self.prefer_libs = set(prefer_libs or ())
        self.ranked: list[str] = []
        self.name = "popularity"

    def fit(self, train_df: pd.DataFrame):
        g = train_df.groupby(KEY_COL if KEY_COL in train_df.columns else LABEL_COL, sort=False)
        counts = g.size()
        if "ingest_lib" in train_df.columns and self.prefer_libs:
            counts = counts * (1 + g["ingest_lib"].agg(
                lambda s: len(set(s.dropna()) & self.prefer_libs)))
        smi = g[LABEL_COL].first()
        self.ranked = smi.loc[counts.sort_values(ascending=False).index].dropna().tolist()
        return self

    def propose(self, q: MoleculeQuery, limit: int = 1000) -> list[str]:
        return self.ranked[:limit]
