"""Peak-list cleaning — the first half of the feature framework.

Every knob the competition's "common curation methods" section mentions lives
on `CleanConfig`, so a cleaning choice is a swappable experiment variable
rather than something buried in a notebook cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

C13 = 1.00335483  # 13C - 12C, the isotope spacing between a peak and its companion


@dataclass(frozen=True)
class CleanConfig:
    """One reproducible preprocessing recipe.

    Defaults are the middle-of-the-road settings named in the data description;
    they are a starting point to beat, not a recommendation.
    """
    min_rel_intensity: float = 0.01     # drop peaks below this fraction of base peak
    drop_above_precursor: bool = True
    precursor_tol_da: float = 1.5       # keep peaks up to precursor + this
    max_peaks: int = 128                # keep the N most intense (0 = keep all)
    window_da: float = 0.0              # >0: keep top `per_window` peaks per window
    per_window: int = 6
    deisotope: bool = False
    deisotope_tol_da: float = 0.01
    intensity_transform: str = "sqrt"   # "none" | "sqrt" | "log" | "rank"
    renormalize: bool = True            # rescale base peak back to 1.0 at the end

    def key(self) -> str:
        """Short stable string for cache filenames and the results table."""
        return "|".join(f"{k}={v}" for k, v in sorted(asdict(self).items()))


def _transform(intensities: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return intensities
    if how == "sqrt":
        return np.sqrt(intensities)
    if how == "log":
        return np.log1p(intensities * 1000.0)
    if how == "rank":
        order = np.argsort(np.argsort(intensities))
        return (order + 1) / len(order) if len(order) else intensities
    raise ValueError(f"unknown intensity_transform {how!r}")


def _window_filter(mzs: np.ndarray, ints: np.ndarray, window: float, per_window: int):
    """Classic library-search filter: top `per_window` peaks per `window` Da.

    Preserves informative low-mass fragments that a global top-N discards
    because the high-mass end of a spectrum is usually more intense.
    """
    if window <= 0 or len(mzs) == 0:
        return mzs, ints
    bucket = np.floor(mzs / window).astype(np.int64)
    keep = np.zeros(len(mzs), dtype=bool)
    order = np.lexsort((-ints, bucket))
    counts: dict[int, int] = {}
    for i in order:
        b = int(bucket[i])
        if counts.get(b, 0) < per_window:
            counts[b] = counts.get(b, 0) + 1
            keep[i] = True
    return mzs[keep], ints[keep]


def _deisotope(mzs: np.ndarray, ints: np.ndarray, tol: float):
    """Remove peaks that look like the 13C companion of a more intense peak."""
    if len(mzs) < 2:
        return mzs, ints
    keep = np.ones(len(mzs), dtype=bool)
    for i in range(len(mzs)):
        if not keep[i]:
            continue
        # a companion sits ~1.0033 Da above and is normally weaker
        lo, hi = mzs[i] + C13 - tol, mzs[i] + C13 + tol
        companions = np.flatnonzero((mzs >= lo) & (mzs <= hi) & (ints <= ints[i]))
        keep[companions] = False
    return mzs[keep], ints[keep]


def clean_spectrum(
    mzs, intensities, precursor_mz: float | None = None, cfg: CleanConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one CleanConfig to a single peak list. Returns m/z-sorted arrays."""
    cfg = cfg or CleanConfig()
    mzs = np.asarray(mzs, dtype=np.float64).ravel()
    ints = np.asarray(intensities, dtype=np.float64).ravel()
    if mzs.size == 0 or mzs.size != ints.size:
        return np.empty(0), np.empty(0)

    finite = np.isfinite(mzs) & np.isfinite(ints) & (ints > 0)
    mzs, ints = mzs[finite], ints[finite]
    if mzs.size == 0:
        return np.empty(0), np.empty(0)

    base = ints.max()
    if base > 0 and cfg.min_rel_intensity > 0:
        keep = ints >= cfg.min_rel_intensity * base
        mzs, ints = mzs[keep], ints[keep]

    if cfg.drop_above_precursor and precursor_mz is not None and np.isfinite(precursor_mz):
        keep = mzs <= precursor_mz + cfg.precursor_tol_da
        mzs, ints = mzs[keep], ints[keep]

    if cfg.deisotope:
        mzs, ints = _deisotope(mzs, ints, cfg.deisotope_tol_da)
    if cfg.window_da > 0:
        mzs, ints = _window_filter(mzs, ints, cfg.window_da, cfg.per_window)
    if cfg.max_peaks and len(mzs) > cfg.max_peaks:
        top = np.argpartition(-ints, cfg.max_peaks)[: cfg.max_peaks]
        mzs, ints = mzs[top], ints[top]

    if mzs.size == 0:
        return np.empty(0), np.empty(0)

    ints = _transform(ints, cfg.intensity_transform)
    if cfg.renormalize and ints.max() > 0:
        ints = ints / ints.max()

    order = np.argsort(mzs)
    return mzs[order], ints[order]


def clean_frame(df, cfg: CleanConfig | None = None, *,
                mz_col="ms2_mzs", int_col="ms2_normalized_intensities",
                prec_col="precursor_mz"):
    """Vectorised-enough wrapper over a spectra DataFrame.

    Adds `mzs_clean`, `ints_clean`, `n_peaks_clean`. Rows left with no peaks are
    kept (not dropped) so row alignment with the caller's frame survives.
    """
    cfg = cfg or CleanConfig()
    cleaned = [
        clean_spectrum(m, i, p, cfg)
        for m, i, p in zip(df[mz_col], df[int_col],
                           df[prec_col] if prec_col in df.columns else [None] * len(df))
    ]
    out = df.copy()
    out["mzs_clean"] = [c[0] for c in cleaned]
    out["ints_clean"] = [c[1] for c in cleaned]
    out["n_peaks_clean"] = [len(c[0]) for c in cleaned]
    return out


# --- spectral similarity ---------------------------------------------------
def align_peaks(mz_a, mz_b, tol_da: float = 0.02, shift: float = 0.0):
    """Greedy one-to-one peak alignment. Returns index pairs (i in a, j in b).

    Greedy-by-closest rather than optimal assignment: it is what matchms and
    most library searches do, and the difference rarely changes a ranking.
    """
    pairs = []
    if len(mz_a) == 0 or len(mz_b) == 0:
        return pairs
    b_shift = np.asarray(mz_b) + shift
    used_b = np.zeros(len(mz_b), dtype=bool)
    for i, m in enumerate(mz_a):
        diffs = np.abs(b_shift - m)
        diffs[used_b] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_da:
            used_b[j] = True
            pairs.append((i, j))
    return pairs


def cosine_similarity(mz_a, int_a, mz_b, int_b, tol_da: float = 0.02) -> float:
    """Plain cosine over aligned peaks."""
    pairs = align_peaks(mz_a, mz_b, tol_da)
    if not pairs:
        return 0.0
    a, b = np.asarray(int_a), np.asarray(int_b)
    num = sum(a[i] * b[j] for i, j in pairs)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num / den) if den > 0 else 0.0


def modified_cosine(mz_a, int_a, prec_a, mz_b, int_b, prec_b, tol_da: float = 0.02) -> float:
    """Modified cosine: also matches peaks offset by the precursor difference.

    This is what lets a library hit survive a mass shift between analogues, and
    it is the single most important similarity for class-2 style lookups.
    """
    a, b = np.asarray(int_a), np.asarray(int_b)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    shift = (prec_a - prec_b) if (np.isfinite(prec_a) and np.isfinite(prec_b)) else 0.0
    direct = align_peaks(mz_a, mz_b, tol_da, shift=0.0)
    shifted = align_peaks(mz_a, mz_b, tol_da, shift=shift) if abs(shift) > tol_da else []

    # one-to-one across both matchings: take the best score per peak pair
    scored = sorted(
        {(i, j): a[i] * b[j] for i, j in list(direct) + list(shifted)}.items(),
        key=lambda kv: -kv[1])
    used_a, used_b, num = set(), set(), 0.0
    for (i, j), val in scored:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); num += val
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num / den) if den > 0 else 0.0
