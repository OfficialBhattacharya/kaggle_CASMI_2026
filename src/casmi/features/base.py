"""Feature framework: a registry of named, swappable molecule representations.

Predictions are made per *molecule*, not per spectrum, so a featurizer's unit of
work is a `MoleculeQuery` — every spectrum of one molecule plus its metadata.
Aggregating inside the featurizer (rather than averaging predictions later) is
deliberate: merging peak lists across collision energies is itself a modelling
choice worth experimenting with.

Add a featurizer:

    @register_featurizer("my_feature")
    class MyFeature(Featurizer):
        def transform(self, q: MoleculeQuery) -> np.ndarray:
            ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from ..chem import neutral_mass_from_precursor
from ..spectra import CleanConfig, clean_spectrum


@dataclass
class MoleculeQuery:
    """All evidence available for one molecule at prediction time."""
    molecule_id: str
    mzs: list[np.ndarray]                 # one cleaned peak list per spectrum
    intensities: list[np.ndarray]
    precursor_mz: np.ndarray
    adduct: list[str]
    ionization_mode: list[str]
    collision_energy_ev: list               # list per spectrum; may be empty
    base_peak_intensity: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_spectra(self) -> int:
        return len(self.mzs)

    def neutral_masses(self) -> np.ndarray:
        """Implied neutral mass per spectrum. NaN where the adduct is unknown."""
        return np.array([neutral_mass_from_precursor(p, a)
                         for p, a in zip(self.precursor_mz, self.adduct)])

    def consensus_mass(self) -> float:
        """Single best neutral-mass estimate, the anchor for candidate lookup."""
        m = self.neutral_masses()
        m = m[np.isfinite(m)]
        return float(np.median(m)) if m.size else float("nan")

    def merged_peaks(self, tol_da: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Union of every spectrum's peaks, intensities summed within `tol_da`."""
        if not self.mzs:
            return np.empty(0), np.empty(0)
        mz = np.concatenate([m for m in self.mzs if len(m)] or [np.empty(0)])
        it = np.concatenate([i for i in self.intensities if len(i)] or [np.empty(0)])
        if mz.size == 0:
            return np.empty(0), np.empty(0)
        order = np.argsort(mz)
        mz, it = mz[order], it[order]
        groups = np.concatenate([[0], np.cumsum(np.diff(mz) > tol_da)])
        n = int(groups[-1]) + 1
        out_mz = np.bincount(groups, weights=mz * it, minlength=n)
        out_it = np.bincount(groups, weights=it, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out_mz = np.where(out_it > 0, out_mz / out_it, 0.0)   # intensity-weighted centroid
        if out_it.max() > 0:
            out_it = out_it / out_it.max()
        return out_mz, out_it


class Featurizer:
    """Base class. Subclasses implement `transform`; `name` labels results."""
    name: str = "unnamed"
    #: set True if the featurizer must see labelled training data first
    needs_fit: bool = False

    def fit(self, queries: Sequence[MoleculeQuery], keys: Sequence[str]) -> "Featurizer":
        return self

    def transform(self, q: MoleculeQuery) -> np.ndarray:
        raise NotImplementedError

    def transform_many(self, queries: Iterable[MoleculeQuery]) -> np.ndarray:
        return np.vstack([self.transform(q) for q in queries])

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


FEATURIZERS: dict[str, Callable[..., Featurizer]] = {}


def register_featurizer(name: str):
    def deco(cls):
        FEATURIZERS[name] = cls
        cls.name = name
        return cls
    return deco


def get_featurizer(name: str, **kwargs) -> Featurizer:
    if name not in FEATURIZERS:
        raise KeyError(f"unknown featurizer {name!r}; have {sorted(FEATURIZERS)}")
    return FEATURIZERS[name](**kwargs)


class FeatureUnion(Featurizer):
    """Concatenate several featurizers — the usual way to build a feature set."""
    def __init__(self, parts: Sequence[Featurizer], name: str | None = None):
        self.parts = list(parts)
        self.name = name or "+".join(p.name for p in self.parts)
        self.needs_fit = any(p.needs_fit for p in self.parts)

    def fit(self, queries, keys):
        for p in self.parts:
            p.fit(queries, keys)
        return self

    def transform(self, q: MoleculeQuery) -> np.ndarray:
        return np.concatenate([np.atleast_1d(p.transform(q)).ravel() for p in self.parts])


# --- building queries from a spectra frame ---------------------------------
def build_queries(
    df: pd.DataFrame,
    cfg: CleanConfig | None = None,
    id_col: str = "molecule_id",
) -> list[MoleculeQuery]:
    """Group a spectra frame into per-molecule queries, cleaning as we go."""
    cfg = cfg or CleanConfig()
    out: list[MoleculeQuery] = []
    for mol_id, g in df.groupby(id_col, sort=False):
        prec = g["precursor_mz"].to_numpy(dtype=float)
        cleaned = [clean_spectrum(m, i, p, cfg) for m, i, p in
                   zip(g["ms2_mzs"], g["ms2_normalized_intensities"], prec)]
        out.append(MoleculeQuery(
            molecule_id=str(mol_id),
            mzs=[c[0] for c in cleaned],
            intensities=[c[1] for c in cleaned],
            precursor_mz=prec,
            adduct=g["adduct"].astype(str).tolist(),
            ionization_mode=g.get("ionization_mode", pd.Series(["?"] * len(g))).astype(str).tolist(),
            collision_energy_ev=g.get("collision_energy_ev", pd.Series([[]] * len(g))).tolist(),
            base_peak_intensity=g.get("base_peak_intensity",
                                      pd.Series([np.nan] * len(g))).to_numpy(dtype=float),
            meta={"spectrum_ids": g.get("spectrum_id", pd.Series(dtype=object)).tolist()},
        ))
    return out
