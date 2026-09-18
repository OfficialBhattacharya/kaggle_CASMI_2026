"""Generate notebooks/02_baseline_submission.ipynb.

Deliberately self-contained: Kaggle code competitions run offline, so this
notebook inlines the handful of functions it needs rather than importing
`casmi`. The repo framework is for local experimentation; once a recipe wins
locally, port the winning pieces here (or attach src/casmi as a Kaggle Dataset,
see scripts/sync_kaggle.py).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md("""
# CASMI 2026 — baseline submission

A first *scored* submission, built to be honest rather than clever. Three stages, deliberately separable:

1. **Candidate generation** — every training structure whose neutral mass matches the query precursor.
2. **Ranking** — spectral library search (modified cosine) over training spectra at a compatible mass.
3. **Padding** — top up to 25 with a natural-product popularity prior, because empty slots are free score.

This can only reach class-1 molecules (and, weakly, class 2 via analogue matching). That is the point:
it establishes the floor the real model has to beat, and tells you what each stage contributes.

Runs offline, CPU only, well under the 9-hour limit.
""")

code("""
import os, re, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

t_start = time.time()
DATA = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
if not DATA.exists():
    DATA = Path(os.environ.get("CASMI_DATA", "../data"))
print("data:", DATA)

# --- knobs (tune these against local CV, not against the public LB) ---
PPM            = 10.0     # mass window for candidate retrieval
LIB_PPM        = 20.0     # mass window for library search
FRAG_TOL_DA    = 0.02     # peak alignment tolerance
MIN_REL_INT    = 0.01     # relative intensity floor
MAX_PEAKS      = 128
INT_TRANSFORM  = "sqrt"
TOP_K          = 25
MAX_LIB_COMPARE= 3000     # cap comparisons per query spectrum (runtime guard)
""")

md("## 1. Chemistry constants")

code("""
PROTON, ELECTRON = 1.00727646677, 0.00054857990
ADDUCT_SHIFT = {
    "[M+H]+": PROTON, "[M+NH4]+": 17.02654910 + PROTON,
    "[M-H2O+H]+": -18.01056468 + PROTON, "[M-2H2O+H]+": -2*18.01056468 + PROTON,
    "[M+Na]+": 22.98976928 - ELECTRON, "[M+K]+": 38.96370649 - ELECTRON,
    "[M-H]-": -PROTON, "[M-H2O-H]-": -18.01056468 - PROTON,
    "[M+CH2O2-H]-": 46.00547931 - PROTON, "[M+Cl]-": 34.96885271 + ELECTRON,
}
def neutral_mass(precursor_mz, adduct):
    s = ADDUCT_SHIFT.get(adduct)
    return np.nan if s is None else precursor_mz - s

EL = {"C":12.0,"H":1.0078250319,"N":14.0030740052,"O":15.9949146221,"S":31.97207069,
      "P":30.97376151,"F":18.000938,"Cl":34.96885271,"Br":78.9183376,"I":126.904468,
      "Si":27.9769265,"Na":22.98976928,"K":38.96370649,"Se":79.9165218,"B":11.0093055}
_RE = re.compile(r"([A-Z][a-z]?)(\\d*)")
def formula_mass(f):
    if not isinstance(f, str): return np.nan
    return sum(EL.get(s, 0.0) * (int(n) if n else 1) for s, n in _RE.findall(f))
""")

md("## 2. Spectrum cleaning")

code("""
def clean(mzs, ints, precursor=None,
          min_rel=MIN_REL_INT, max_peaks=MAX_PEAKS, transform=INT_TRANSFORM):
    mzs  = np.asarray(mzs,  dtype=np.float64).ravel()
    ints = np.asarray(ints, dtype=np.float64).ravel()
    if mzs.size == 0 or mzs.size != ints.size:
        return np.empty(0), np.empty(0)
    ok = np.isfinite(mzs) & np.isfinite(ints) & (ints > 0)
    mzs, ints = mzs[ok], ints[ok]
    if mzs.size == 0: return np.empty(0), np.empty(0)
    if min_rel > 0:
        keep = ints >= min_rel * ints.max(); mzs, ints = mzs[keep], ints[keep]
    if precursor is not None and np.isfinite(precursor):
        keep = mzs <= precursor + 1.5; mzs, ints = mzs[keep], ints[keep]
    if mzs.size == 0: return np.empty(0), np.empty(0)
    if max_peaks and mzs.size > max_peaks:
        top = np.argpartition(-ints, max_peaks)[:max_peaks]; mzs, ints = mzs[top], ints[top]
    if transform == "sqrt": ints = np.sqrt(ints)
    elif transform == "log": ints = np.log1p(ints * 1000.0)
    if ints.max() > 0: ints = ints / ints.max()
    o = np.argsort(mzs)
    return mzs[o], ints[o]

def align(mz_a, mz_b, tol, shift=0.0):
    pairs, b = [], np.asarray(mz_b) + shift
    used = np.zeros(len(mz_b), dtype=bool)
    for i, m in enumerate(mz_a):
        d = np.abs(b - m); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True; pairs.append((i, j))
    return pairs

def modified_cosine(mz_a, in_a, pr_a, mz_b, in_b, pr_b, tol=FRAG_TOL_DA):
    \"\"\"Cosine that also matches peaks offset by the precursor difference,
    so an analogue with a shifted backbone still scores.\"\"\"
    a, b = np.asarray(in_a), np.asarray(in_b)
    if a.size == 0 or b.size == 0: return 0.0
    shift = pr_a - pr_b if np.isfinite(pr_a) and np.isfinite(pr_b) else 0.0
    cand = {(i, j): a[i]*b[j] for i, j in align(mz_a, mz_b, tol)}
    if abs(shift) > tol:
        cand.update({(i, j): a[i]*b[j] for i, j in align(mz_a, mz_b, tol, shift)})
    ua, ub, num = set(), set(), 0.0
    for (i, j), v in sorted(cand.items(), key=lambda kv: -kv[1]):
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j); num += v
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(num/den) if den > 0 else 0.0
""")

md("## 3. Load")

code("""
%%time
test = pq.read_table(DATA / "test.parquet").to_pandas()
print(f"test: {len(test):,} spectra / {test.molecule_id.nunique()} molecules")

TRAIN_COLS = ["ms2_mzs","ms2_normalized_intensities","precursor_mz","adduct",
              "normalized_smiles","inchikey14","molecular_formula","ingest_lib"]
train = pq.read_table(DATA / "train.parquet", columns=TRAIN_COLS).to_pandas()
print(f"train: {len(train):,} spectra / {train.inchikey14.nunique():,} structures")
""")

md("""
## 4. Candidate database

Unique training structures indexed by neutral mass. Popularity is up-weighted for structures seen in
natural-product libraries, since the test set is natural products and `enveda-180` (46% of train) is not.

**This is the ceiling on class 2.** Replacing this table with a PubChem/COCONUT subset — attached as a
Kaggle Dataset, which is allowed — is the single highest-value upgrade to this baseline.
""")

code("""
%%time
NP_LIBS = {"enveda-np-examples","riken","gnps","spectraverse","massbank"}
g = train.groupby("inchikey14", sort=False)
struct = pd.DataFrame({
    "smiles": g.normalized_smiles.first(),
    "formula": g.molecular_formula.first(),
    "n_spectra": g.size(),
    "np_libs": g.ingest_lib.agg(lambda s: len(set(s.dropna()) & NP_LIBS)),
}).reset_index()
struct["popularity"] = struct.n_spectra * (1 + struct.np_libs)
struct["mass"] = struct.formula.map(formula_mass)
struct = struct[struct.mass.between(50, 2000)].sort_values("mass").reset_index(drop=True)
STRUCT_MASS = struct.mass.to_numpy()
print(f"{len(struct):,} indexed structures, mass {STRUCT_MASS.min():.0f}-{STRUCT_MASS.max():.0f} Da")

POPULAR = struct.sort_values("popularity", ascending=False).smiles.head(200).tolist()

def candidates_by_mass(mass, ppm=PPM, limit=2000):
    if not np.isfinite(mass): return []
    tol = mass * ppm * 1e-6
    lo, hi = np.searchsorted(STRUCT_MASS, [mass - tol, mass + tol])
    if hi <= lo: return []
    sl = struct.iloc[lo:hi].sort_values("popularity", ascending=False)
    return sl.smiles.head(limit).tolist()
""")

md("## 5. Spectral library index")

code("""
%%time
# Index every training spectrum by NEUTRAL mass, not precursor m/z: the same molecule
# measured as [M+Na]+ sits ~22 Da from its [M+H]+ spectrum and any ppm window on
# precursor m/z would hide it.
tr = train.copy()
tr["neutral"] = [neutral_mass(p, a) for p, a in zip(tr.precursor_mz, tr.adduct)]
tr = tr[np.isfinite(tr.neutral) & tr.normalized_smiles.notna()].sort_values("neutral")

LIB_MASS = tr.neutral.to_numpy()
LIB_PREC = tr.precursor_mz.to_numpy()
LIB_SMI  = tr.normalized_smiles.to_numpy()
LIB_KEY  = tr.inchikey14.to_numpy()
LIB_MZ   = tr.ms2_mzs.to_numpy()
LIB_INT  = tr.ms2_normalized_intensities.to_numpy()
print(f"library: {len(tr):,} searchable spectra")
del tr; gc.collect()
""")

md("## 6. Rank each test molecule")

code("""
%%time
test_g = test.groupby("molecule_id", sort=False)
predictions, diag = {}, []

for n_done, (mol_id, grp) in enumerate(test_g, 1):
    prec = grp.precursor_mz.to_numpy(dtype=float)
    masses = np.array([neutral_mass(p, a) for p, a in zip(prec, grp.adduct)])
    finite = masses[np.isfinite(masses)]
    mass = float(np.median(finite)) if finite.size else np.nan

    q = [clean(m, i, p) for m, i, p in
         zip(grp.ms2_mzs, grp.ms2_normalized_intensities, prec)]

    # --- stage 2: library search, aggregated across this molecule's spectra ---
    best = {}
    if np.isfinite(mass):
        tol = mass * LIB_PPM * 1e-6 + 0.01
        lo, hi = np.searchsorted(LIB_MASS, [mass - tol, mass + tol])
        idx = np.arange(lo, min(hi, lo + MAX_LIB_COMPARE))
        for (qm, qi), qp in zip(q, prec):
            if qm.size == 0: continue
            for j in idx:
                lm, li = clean(LIB_MZ[j], LIB_INT[j], LIB_PREC[j])
                if lm.size == 0: continue
                s = modified_cosine(qm, qi, qp, lm, li, LIB_PREC[j])
                k = LIB_KEY[j]
                if s > best.get(k, (0.0, None))[0]:
                    best[k] = (s, LIB_SMI[j])

    ranked = [smi for _, smi in sorted(best.values(), key=lambda t: -t[0])][:TOP_K]
    n_lib = len(ranked)

    # --- stage 1+3: pad with mass-window candidates, then global popularity ---
    seen = set(ranked)
    for smi in candidates_by_mass(mass) + POPULAR:
        if len(ranked) >= TOP_K: break
        if smi not in seen:
            seen.add(smi); ranked.append(smi)

    predictions[mol_id] = ranked
    diag.append({"molecule_id": mol_id, "n_spectra": len(grp), "mass": mass,
                 "n_from_library": n_lib, "n_final": len(ranked)})
    if n_done % 50 == 0:
        print(f"  {n_done}/{test.molecule_id.nunique()} molecules  "
              f"({time.time()-t_start:.0f}s elapsed)", flush=True)

diag = pd.DataFrame(diag)
print(f"\\ndone in {time.time()-t_start:.0f}s")
""")

code("""
print("how many of the 25 slots came from spectral matching vs padding:")
display(diag.n_from_library.describe(percentiles=[.25,.5,.75]).round(1).to_frame().T)
print(f"molecules with NO library hit at all: {(diag.n_from_library==0).sum()} "
      f"({(diag.n_from_library==0).mean():.1%}) -- these rely entirely on the mass prior")
print(f"molecules with no usable neutral mass: {diag.mass.isna().sum()}")
""")

md("## 7. Write and validate the submission")

code("""
MAX_GUESSES = 25
rows = []
for mol_id in sorted(test.molecule_id.unique()):
    g = [str(s) for s in predictions.get(mol_id, []) if s and str(s).strip()]
    seen, uniq = set(), []
    for s in g:                      # a duplicate guess wastes a ranked slot
        if s not in seen: seen.add(s); uniq.append(s)
    if not uniq: uniq = POPULAR[:MAX_GUESSES] or ["C"]
    rows.append({"molecule_id": mol_id, "smiles": ";".join(uniq[:MAX_GUESSES])})
submission = pd.DataFrame(rows)

# Fail loudly here rather than after a 9-hour run.
assert list(submission.columns) == ["molecule_id", "smiles"], submission.columns
assert len(submission) > 0
assert submission.molecule_id.is_unique, "repeated molecule_id"
assert submission.notna().all().all(), "nulls present"
assert (submission.smiles.str.strip() != "").all(), "blank smiles"
assert (submission.smiles.str.count(";") + 1 <= MAX_GUESSES).all(), ">25 guesses"
assert set(submission.molecule_id) == set(test.molecule_id.astype(str)), "molecule_id mismatch"

submission.to_csv("submission.csv", index=False)
print(f"submission.csv written: {len(submission)} rows")
print(f"mean guesses per row: {(submission.smiles.str.count(';')+1).mean():.1f} / 25")
print(f"total notebook time: {time.time()-t_start:.0f}s")
display(submission.head(3))
""")

md("""
## What to improve, in order of expected payoff

1. **Attach an external structure database** (PubChem / COCONUT as a Kaggle Dataset). Class 2 is
   currently unreachable — the candidate pool is only structures that already have training spectra.
   This is the biggest single gain available.
2. **Predict the molecular formula first** (SIRIUS / MIST-CF style) and filter candidates by it.
   A formula constraint is far tighter than a ppm mass window and cuts the decoy count hard.
3. **Learn the ranker.** Replace raw modified cosine with a model scoring (spectrum, candidate)
   pairs — fingerprint prediction from the spectrum, then Tanimoto against candidate fingerprints,
   is the established strong approach.
4. **Domain-adapt to timsTOF** using `enveda-np-examples`, the only library that matches the test set
   on both instrument and chemistry.
5. **De novo generation** for class 3. Nothing else can touch it, and it is a hard cap on the score.

Measure each of these locally with `casmi.experiment` before spending a submission on it —
the per-class breakdown tells you which stage a change actually moved.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_baseline_submission.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells)")
