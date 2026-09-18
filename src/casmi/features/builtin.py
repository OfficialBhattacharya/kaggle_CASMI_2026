"""Built-in featurizers. Each one is an experiment variable you can swap."""
from __future__ import annotations

import numpy as np

from ..chem import ADDUCT_MASS_SHIFT
from ..config import TEST_ADDUCTS
from .base import Featurizer, MoleculeQuery, register_featurizer

_ADDUCT_INDEX = {a: i for i, a in enumerate(TEST_ADDUCTS)}


@register_featurizer("binned")
class BinnedSpectrum(Featurizer):
    """Fixed-width m/z histogram of the merged peak list — the classic baseline.

    Coarse bins blur isobaric fragments; fine bins make the vector sparse and
    tolerance-sensitive. Worth sweeping `bin_width` first of anything here.
    """
    def __init__(self, bin_width: float = 1.0, max_mz: float = 1200.0, merge_tol: float = 0.01):
        self.bin_width, self.max_mz, self.merge_tol = bin_width, max_mz, merge_tol
        self.n_bins = int(np.ceil(max_mz / bin_width))
        self.name = f"binned(w={bin_width},max={max_mz:g})"

    def transform(self, q: MoleculeQuery) -> np.ndarray:
        mz, it = q.merged_peaks(self.merge_tol)
        v = np.zeros(self.n_bins, dtype=np.float32)
        if mz.size:
            keep = (mz >= 0) & (mz < self.max_mz)
            idx = (mz[keep] / self.bin_width).astype(int)
            np.add.at(v, idx, it[keep])
        m = v.max()
        return v / m if m > 0 else v


@register_featurizer("neutral_loss")
class NeutralLossSpectrum(Featurizer):
    """Histogram of precursor - fragment, i.e. the mass of what was lost.

    Complementary to `binned`: losses are often conserved across a series of
    analogues whose absolute fragment masses all differ.
    """
    def __init__(self, bin_width: float = 1.0, max_loss: float = 1000.0):
        self.bin_width, self.max_loss = bin_width, max_loss
        self.n_bins = int(np.ceil(max_loss / bin_width))
        self.name = f"neutral_loss(w={bin_width})"

    def transform(self, q: MoleculeQuery) -> np.ndarray:
        v = np.zeros(self.n_bins, dtype=np.float32)
        for mz, it, prec in zip(q.mzs, q.intensities, q.precursor_mz):
            if not len(mz) or not np.isfinite(prec):
                continue
            loss = prec - mz
            keep = (loss >= 0) & (loss < self.max_loss)
            if keep.any():
                np.add.at(v, (loss[keep] / self.bin_width).astype(int), it[keep])
        m = v.max()
        return v / m if m > 0 else v


@register_featurizer("topn_peaks")
class TopNPeaks(Featurizer):
    """Flat [mz, intensity] pairs of the N most intense merged peaks.

    Keeps exact m/z rather than binning it — the input shape a transformer or
    set-based model wants. Zero-padded to a fixed length.
    """
    def __init__(self, n: int = 64, scale_mz: float = 1000.0):
        self.n, self.scale_mz = n, scale_mz
        self.name = f"topn_peaks(n={n})"

    def transform(self, q: MoleculeQuery) -> np.ndarray:
        mz, it = q.merged_peaks()
        out = np.zeros((self.n, 2), dtype=np.float32)
        if mz.size:
            top = np.argsort(-it)[: self.n]
            top = top[np.argsort(mz[top])]
            out[: len(top), 0] = mz[top] / self.scale_mz
            out[: len(top), 1] = it[top]
        return out.ravel()


@register_featurizer("meta")
class MetaFeatures(Featurizer):
    """Non-peak evidence: mass, adduct, polarity, collision energy, peak counts.

    Cheap and always worth including — precursor mass alone constrains the
    formula hard enough to carry a surprising amount of a baseline's score.
    """
    def transform(self, q: MoleculeQuery) -> np.ndarray:
        mass = q.consensus_mass()
        prec = q.precursor_mz[np.isfinite(q.precursor_mz)]
        n_peaks = np.array([len(m) for m in q.mzs], dtype=float)

        ces = [float(np.mean(c)) for c in q.collision_energy_ev
               if c is not None and np.ndim(c) and len(c)]
        ce = np.array(ces) if ces else np.array([np.nan])

        adducts = np.zeros(len(TEST_ADDUCTS) + 1, dtype=np.float32)
        for a in q.adduct:
            adducts[_ADDUCT_INDEX.get(a, len(TEST_ADDUCTS))] += 1
        if adducts.sum():
            adducts /= adducts.sum()

        bpi = q.base_peak_intensity[np.isfinite(q.base_peak_intensity)]
        scalars = np.array([
            mass if np.isfinite(mass) else 0.0,
            mass / 1000.0 if np.isfinite(mass) else 0.0,
            prec.min() if prec.size else 0.0,
            prec.max() if prec.size else 0.0,
            q.n_spectra,
            n_peaks.mean() if n_peaks.size else 0.0,
            n_peaks.max() if n_peaks.size else 0.0,
            np.nan_to_num(np.nanmean(ce)),
            np.nan_to_num(np.nanmin(ce)),
            np.nan_to_num(np.nanmax(ce)),
            float(any(m == "positive" for m in q.ionization_mode)),
            float(any(m == "negative" for m in q.ionization_mode)),
            np.log1p(bpi.mean()) if bpi.size else 0.0,
        ], dtype=np.float32)
        return np.concatenate([scalars, adducts])


@register_featurizer("peak_stats")
class PeakStats(Featurizer):
    """Shape descriptors of the merged peak list: entropy, spread, mass defect.

    Spectral entropy is a well-established quality/complexity signal, and the
    fractional mass of the precursor is a weak but free formula constraint.
    """
    def transform(self, q: MoleculeQuery) -> np.ndarray:
        mz, it = q.merged_peaks()
        if not mz.size:
            return np.zeros(9, dtype=np.float32)
        p = it / it.sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        mass = q.consensus_mass()
        mass = mass if np.isfinite(mass) else 0.0
        prec = float(np.nanmedian(q.precursor_mz))
        return np.array([
            len(mz), entropy, entropy / np.log(len(mz) + 1e-12) if len(mz) > 1 else 0.0,
            mz.min(), mz.max(), float(np.average(mz, weights=it)), mz.std(),
            mass - np.floor(mass),                       # mass defect
            (mz.max() / prec) if prec > 0 else 0.0,      # how close to the precursor
        ], dtype=np.float32)
