"""Built-in rankers: the baselines every later model has to beat."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..candidates.base import CandidateSource
from ..config import KEY_COL, LABEL_COL, MAX_GUESSES
from ..features.base import Featurizer, MoleculeQuery, build_queries
from ..spectra import CleanConfig, cosine_similarity, modified_cosine
from .base import Ranker, register_model


@register_model("candidate_prior")
class CandidatePriorRanker(Ranker):
    """Rank purely by the candidate source's own prior — no spectrum matching.

    Deliberately weak. It measures how much score comes from the precursor mass
    plus a popularity prior alone, which is the number every spectral model must
    be judged against: beating zero is easy, beating this is the real bar.
    """
    def __init__(self, source: CandidateSource):
        self.source = source
        self.name = f"candidate_prior({source.name})"

    def fit(self, train_df, featurizer=None):
        self.source.fit(train_df)
        return self

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        return self.source.propose(q, k)


@register_model("library_search")
class LibrarySearchRanker(Ranker):
    """Spectral library search — the strong class-1 baseline.

    For each query spectrum, compare against training spectra whose precursor
    falls in the same mass window, then aggregate per structure across all of
    the molecule's spectra.

    `similarity`:  "cosine" matches only identical structures;
                   "modified" also matches analogues shifted by the precursor
                   difference, which is what reaches class-2 molecules.
    `aggregate`:   how a structure's several library spectra combine — "max" is
                   the classic library hit, "mean" is steadier when the library
                   holds many collision energies of the same compound.
    """
    def __init__(self, ppm: float = 20.0, tol_da: float = 0.02, similarity: str = "modified",
                 aggregate: str = "max", max_compare: int = 4000,
                 clean: CleanConfig | None = None, min_similarity: float = 0.0,
                 analogue_window_da: float = 250.0):
        self.ppm, self.tol_da = ppm, tol_da
        self.analogue_window_da = analogue_window_da
        self.similarity, self.aggregate = similarity, aggregate
        self.max_compare, self.min_similarity = max_compare, min_similarity
        self.clean = clean or CleanConfig()
        self.name = f"library_search({similarity},ppm={ppm},agg={aggregate})"
        self._mass = np.empty(0)
        self._lib: list[tuple] = []

    def fit(self, train_df: pd.DataFrame, featurizer: Featurizer | None = None):
        """Index every training spectrum by *neutral* mass.

        Indexing on precursor m/z would hide the same molecule measured under a
        different adduct — an [M+Na]+ library spectrum sits ~22 Da away from the
        [M+H]+ query and falls outside any sane ppm window. Neutral mass puts
        every adduct of a structure at the same coordinate. The modified-cosine
        peak shift still uses the raw precursor difference, which is what makes
        fragments comparable across adducts.

        Spectra whose adduct is unknown keep their precursor m/z as a stand-in,
        so they stay searchable rather than being silently dropped.
        """
        from ..chem import neutral_mass_from_precursor
        from ..spectra import clean_spectrum
        df = train_df[train_df[LABEL_COL].notna()]
        recs = []
        for mz, it, prec, adduct, smi, key in zip(
            df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"],
            df["adduct"].astype(str),
            df[LABEL_COL], df[KEY_COL] if KEY_COL in df.columns else df[LABEL_COL],
        ):
            cm, ci = clean_spectrum(mz, it, prec, self.clean)
            if not len(cm):
                continue
            mass = neutral_mass_from_precursor(float(prec), adduct)
            if not np.isfinite(mass):
                mass = float(prec)
            recs.append((mass, float(prec), cm, ci, smi, key))
        recs.sort(key=lambda r: r[0])
        self._mass = np.array([r[0] for r in recs])
        self._lib = recs
        return self

    def _window(self, mass: float) -> range:
        """Library rows whose neutral mass is compatible with the query."""
        if self._mass.size == 0 or not np.isfinite(mass):
            return range(0)
        if self.similarity == "modified":
            # analogues sit at a different mass entirely, so widen the window
            lo, hi = mass - self.analogue_window_da, mass + self.analogue_window_da
        else:
            tol = mass * self.ppm * 1e-6 + 0.01
            lo, hi = mass - tol, mass + tol
        a, b = np.searchsorted(self._mass, [lo, hi])
        return range(int(a), int(b))

    def _structure_scores(self, q: MoleculeQuery) -> dict[str, tuple[float, str]]:
        per_struct: dict[str, list[float]] = {}
        smiles_of: dict[str, str] = {}
        masses = q.neutral_masses()
        for mz, it, prec, mass in zip(q.mzs, q.intensities, q.precursor_mz, masses):
            if not len(mz):
                continue
            if not np.isfinite(mass):
                mass = float(prec)
            idx = list(self._window(mass))
            if len(idx) > self.max_compare:      # keep the nearest by mass
                centre = int(np.searchsorted(self._mass, mass))
                half = self.max_compare // 2
                idx = list(range(max(0, centre - half),
                                 min(len(self._lib), centre + half)))
            for i in idx:
                lmass, lprec, lmz, lit, lsmi, lkey = self._lib[i]
                sim = (modified_cosine(mz, it, prec, lmz, lit, lprec, self.tol_da)
                       if self.similarity == "modified"
                       else cosine_similarity(mz, it, lmz, lit, self.tol_da))
                if sim > self.min_similarity:
                    per_struct.setdefault(lkey, []).append(sim)
                    smiles_of.setdefault(lkey, lsmi)
        agg = {"max": max, "mean": lambda v: float(np.mean(v)),
               "sum": sum}[self.aggregate]
        return {k: (agg(v), smiles_of[k]) for k, v in per_struct.items()}

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        scored = self._structure_scores(q)
        best = sorted(scored.values(), key=lambda t: -t[0])[:k]
        return [smi for _, smi in best]

    def score_candidates(self, q: MoleculeQuery, candidates) -> np.ndarray:
        from ..chem import KEY_RESOLVER
        scored = self._structure_scores(q)
        by_key = {k: s for k, (s, _) in scored.items()}
        return np.array([by_key.get(KEY_RESOLVER(c), 0.0) for c in candidates])


@register_model("feature_knn")
class FeatureKNNRanker(Ranker):
    """k-NN in a featurizer's vector space — the harness for feature experiments.

    This is the model you hold fixed while sweeping featurizers: it has almost
    no capacity of its own, so a change in score is a change in the features.
    Cosine over L2-normalised vectors, restricted to a precursor-mass window.
    """
    def __init__(self, k_neighbours: int = 50, ppm: float = 20.0,
                 mass_filter: bool = True, aggregate: str = "max"):
        self.k_neighbours, self.ppm = k_neighbours, ppm
        self.mass_filter, self.aggregate = mass_filter, aggregate
        self.name = f"feature_knn(k={k_neighbours},mass_filter={mass_filter})"
        self.X = np.empty((0, 0))
        self.keys: list[str] = []
        self.smiles: list[str] = []
        self.mass = np.empty(0)
        self.featurizer: Featurizer | None = None

    def fit(self, train_df: pd.DataFrame, featurizer: Featurizer | None = None):
        if featurizer is None:
            raise ValueError("feature_knn requires a featurizer — it is the thing under test")
        self.featurizer = featurizer
        df = train_df[train_df[LABEL_COL].notna()]
        queries = build_queries(df, id_col="spectrum_id" if "spectrum_id" in df else "molecule_id")
        lookup = df.set_index("spectrum_id" if "spectrum_id" in df else "molecule_id")
        X = featurizer.transform_many(queries)
        X = np.nan_to_num(X.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = X / np.where(norms > 0, norms, 1.0)
        self.smiles = [lookup.loc[q.molecule_id, LABEL_COL] for q in queries]
        key_col = KEY_COL if KEY_COL in df.columns else LABEL_COL
        self.keys = [lookup.loc[q.molecule_id, key_col] for q in queries]
        self.mass = np.array([q.consensus_mass() for q in queries])
        return self

    def rank(self, q: MoleculeQuery, k: int = MAX_GUESSES) -> list[str]:
        if self.X.size == 0 or self.featurizer is None:
            return []
        v = np.nan_to_num(self.featurizer.transform(q).ravel().astype(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            return []
        sims = self.X @ (v / n)

        if self.mass_filter:
            mass = q.consensus_mass()
            if np.isfinite(mass):
                tol = mass * self.ppm * 1e-6 + 0.01
                sims = np.where(np.abs(self.mass - mass) <= tol, sims, -np.inf)

        top = np.argsort(-sims)[: max(self.k_neighbours, k * 4)]
        top = top[np.isfinite(sims[top])]
        best: dict[str, tuple[float, str]] = {}
        for i in top:
            key, s = self.keys[i], float(sims[i])
            if key not in best or s > best[key][0]:
                best[key] = (s, self.smiles[i])
        return [smi for _, smi in sorted(best.values(), key=lambda t: -t[0])[:k]]
