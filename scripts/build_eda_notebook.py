"""Generate notebooks/01_eda.ipynb.

The notebook is built from this script rather than edited by hand so it stays
diffable in git — a committed .ipynb full of execution counts and output blobs
is unreviewable, and this project is meant to be worked from several devices.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

MD, CODE = [], []
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# CASMI 2026 — EDA: what the data is, what the task is, what to submit

**Goal of this notebook:** understand the problem well enough to design a model, before writing one.

The task in one sentence: given the tandem mass spectra of an unknown molecule, output **up to 25 candidate
SMILES strings, ranked best-first**, and you are scored on where the correct structure lands.

Three things make this different from a normal Kaggle tabular problem, and all three shape everything downstream:

1. **The output is a ranked list of molecules, not a number or a class.** There is no fixed label set —
   the answer space is "all of chemistry". So the real pipeline is *candidate generation* → *ranking*,
   and those two stages fail in completely different ways.
2. **Prediction is per molecule, not per spectrum.** A molecule may appear as several spectra
   (different collision energies, different adducts). You must aggregate them into one ranked list.
3. **A third of the test set is, by design, not in any database.** The hidden test mixes three
   *novelty classes*, and the mix is hidden for the whole competition. A method that nails class 1
   and scores zero on class 3 has a hard ceiling.

| Class | Definition | What can possibly solve it |
|---|---|---|
| 1 | Public reference spectra exist | spectral library search |
| 2 | Structure in PubChem/COCONUT, no spectra | database retrieval + reranking |
| 3 | Not in PubChem at all | *de novo* structure generation |
""")

md("## 0. Setup")

code("""
import os, sys, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)

DATA = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
if not DATA.exists():                      # running locally
    DATA = Path(os.environ.get("CASMI_DATA", "../data"))
print("data dir:", DATA, "|", sorted(p.name for p in DATA.glob("*")))

try:
    import matplotlib.pyplot as plt
    PLOT = True
    plt.rcParams.update({"figure.figsize": (9, 3.4), "axes.grid": True,
                         "grid.alpha": .3, "font.size": 9})
except ImportError:
    PLOT = False
    print("matplotlib unavailable — tables only")
""")

md("""
### Loading strategy

`train.parquet` is ~2.5M spectra with two array columns; loading it whole is slow and memory-hungry.
Read the metadata columns first — they answer most EDA questions on their own — and only pull the
peak arrays for the subset actually being plotted.
""")

code("""
import pyarrow.parquet as pq

train_schema = pq.read_schema(DATA / "train.parquet")
test_schema  = pq.read_schema(DATA / "test.parquet")
print("train columns:", train_schema.names)
print("\\ntest  columns:", test_schema.names)
print("\\ntrain-only  :", sorted(set(train_schema.names) - set(test_schema.names)))

PEAK_COLS = ["ms2_mzs", "ms2_normalized_intensities"]
META_COLS = [c for c in train_schema.names if c not in PEAK_COLS]
""")

code("""
%%time
# Metadata only — this is what almost every cell below needs.
train = pq.read_table(DATA / "train.parquet", columns=META_COLS).to_pandas()
test  = pq.read_table(DATA / "test.parquet").to_pandas()   # small enough to take whole
sub   = pd.read_csv(DATA / "sample_submission.csv")

print(f"train : {len(train):,} spectra")
print(f"test  : {len(test):,} spectra")
print(f"sample_submission : {sub.shape}")
print(f"\\ntrain memory (metadata only): {train.memory_usage(deep=True).sum()/1e9:.2f} GB")
""")

md("## 1. What one row looks like")

code("""
print("=== a single TEST row ===")
row = test.iloc[0]
for c in test.columns:
    v = row[c]
    if isinstance(v, np.ndarray):
        print(f"{c:30s} array(len={len(v)})  head={np.round(v[:5], 4).tolist()}")
    else:
        print(f"{c:30s} {v!r}")
""")

code("""
print("=== the extra columns TRAIN has ===")
train[sorted(set(train.columns) - set(test.columns))].head(3).T
""")

md("""
### Reading a spectrum

`ms2_mzs` and `ms2_normalized_intensities` are **aligned arrays**: the i-th m/z pairs with the i-th
intensity. Intensities are normalised per spectrum so the base peak equals 1.0.
""")

code("""
# Pull the peak arrays for just a handful of test spectra.
peek = pq.read_table(DATA / "test.parquet", columns=PEAK_COLS + ["spectrum_id","molecule_id","precursor_mz","adduct","collision_energy_ev"]).to_pandas().head(400)

ex = peek.iloc[0]
print(f"molecule {ex.molecule_id} | {ex.adduct} | precursor m/z {ex.precursor_mz:.4f} | CE {ex.collision_energy_ev}")
print(f"{len(ex.ms2_mzs)} peaks")
pd.DataFrame({"mz": ex.ms2_mzs, "intensity": ex.ms2_normalized_intensities}).sort_values("intensity", ascending=False).head(10)
""")

code("""
if PLOT:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, i in zip(axes, [0, 1]):
        r = peek.iloc[i]
        ax.vlines(r.ms2_mzs, 0, r.ms2_normalized_intensities, lw=.9)
        ax.axvline(r.precursor_mz, color="crimson", ls="--", lw=1,
                   label=f"precursor {r.precursor_mz:.3f}")
        ax.set_ylabel("rel. intensity")
        ax.set_title(f"{r.molecule_id} · {r.adduct} · CE {r.collision_energy_ev} · {len(r.ms2_mzs)} peaks", fontsize=9)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("m/z")
    plt.tight_layout(); plt.show()
""")

md("""
## 2. The test set: molecules vs spectra

This is the first thing to internalise — **you are scored per molecule, not per spectrum.**
""")

code("""
per_mol = test.groupby("molecule_id").size()
print(f"test spectra   : {len(test):,}")
print(f"test molecules : {test.molecule_id.nunique():,}")
print(f"spectra per molecule: min={per_mol.min()} median={per_mol.median():.0f} max={per_mol.max()}")
print(f"\\nmolecules with only 1 spectrum: {(per_mol==1).sum()} ({(per_mol==1).mean():.1%})")
print(f"\\nsubmission rows == test molecules? {len(sub) == test.molecule_id.nunique()}")
display(per_mol.value_counts().sort_index().rename("molecules").to_frame().T)
""")

code("""
# Does a molecule's spectra vary in adduct / energy? If so, they carry complementary information.
g = test.groupby("molecule_id")
multi = per_mol[per_mol > 1].index
print(f"of {len(multi)} multi-spectrum molecules:")
print(f"  vary in adduct        : {g.adduct.nunique().loc[multi].gt(1).sum()}")
print(f"  vary in ion mode      : {g.ionization_mode.nunique().loc[multi].gt(1).sum()}")
ce_str = test.assign(ce=test.collision_energy_ev.astype(str))
print(f"  vary in collision E   : {ce_str.groupby('molecule_id').ce.nunique().loc[multi].gt(1).sum()}")
""")

md("""
## 3. The label, and what "correct" actually means

The label is `normalized_smiles`, but **grading does not compare SMILES strings.** Both your guess and
the answer are tautomer-canonicalised and reduced to the **first block of the InChIKey** (`inchikey14`),
which encodes 2D atom connectivity only.

Practical consequences:
- Stereochemistry is **free** — don't spend model capacity on it.
- Tautomers are **free**.
- `inchikey14` is the unit of everything: dedup on it, evaluate on it, and never on raw SMILES.
""")

code("""
print(f"unique normalized_smiles : {train.normalized_smiles.nunique():,}")
print(f"unique inchikey (full)   : {train.inchikey.nunique():,}")
print(f"unique inchikey14 (2D)   : {train.inchikey14.nunique():,}   <-- the grading unit")
collapse = 1 - train.inchikey14.nunique() / train.normalized_smiles.nunique()
print(f"\\n{collapse:.1%} of distinct SMILES collapse into another structure under inchikey14")
print("(stereoisomers that grading treats as the same answer)")
""")

code("""
spec_per_struct = train.groupby("inchikey14").size()
print("spectra per structure in train:")
print(spec_per_struct.describe(percentiles=[.5,.9,.99]).round(1).to_string())
print(f"\\nstructures with a single spectrum: {(spec_per_struct==1).sum():,} "
      f"({(spec_per_struct==1).mean():.1%})")
""")

md("""
## 4. The libraries — the single most important strategic fact

`train.parquet` is an aggregation of 11 sources with different instruments, protocols and *chemistry*.
Compare each library against the test set, which is **100% Bruker timsTOF** and **natural products**.
""")

code("""
lib = train.groupby("ingest_lib").agg(
    spectra=("spectrum_id", "size"),
    structures=("inchikey14", "nunique"),
    median_prec=("precursor_mz", "median"),
    pct_positive=("ionization_mode", lambda s: (s == "positive").mean()),
).sort_values("spectra", ascending=False)
lib["pct_of_train"] = lib.spectra / lib.spectra.sum()
lib["spectra_per_structure"] = (lib.spectra / lib.structures).round(1)
display(lib.style.format({"spectra":"{:,}", "structures":"{:,}", "median_prec":"{:.0f}",
                          "pct_positive":"{:.0%}", "pct_of_train":"{:.1%}"}))
""")

code("""
# instrument_type is free text in train, a single value in test.
print("test instrument_type:", test.instrument_type.unique())
print(f"\\ntrain instrument_type: {train.instrument_type.nunique()} distinct strings, "
      f"{train.instrument_type.isna().mean():.1%} null")
display(train.instrument_type.value_counts(dropna=False).head(12).rename("spectra").to_frame())

timstof = train.instrument_type.fillna("").str.contains("timsTOF", case=False)
print(f"\\nspectra whose instrument string mentions timsTOF: {timstof.sum():,} ({timstof.mean():.1%})")
""")

md("""
> **The core tension.** `enveda-180` is 46% of the training data and is on the *right instrument*,
> but it is synthetic drug-like screening compounds — the wrong chemistry. The natural-product
> libraries (`riken`, `gnps`, `spectraverse`, `massbank`) are the *right chemistry* on the
> *wrong instruments*. `enveda-np-examples` is right on both counts and has only 1,151 spectra.
>
> Every design decision downstream is some version of trading these off. Validating on a uniform
> random sample of train validates on the wrong distribution.
""")

md("## 5. Adducts and ionization mode — train vs test")

code("""
TEST_ADDUCTS = ["[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+",
                "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"]

cmp_ = pd.DataFrame({
    "test_pct": test.adduct.value_counts(normalize=True),
    "train_pct": train.adduct.value_counts(normalize=True),
}).loc[lambda d: d.index.isin(TEST_ADDUCTS) | (d.test_pct.notna())]
cmp_["train_spectra"] = train.adduct.value_counts()
display(cmp_.sort_values("test_pct", ascending=False).head(15)
        .style.format({"test_pct":"{:.1%}", "train_pct":"{:.1%}", "train_spectra":"{:,.0f}"}))

print(f"train has {train.adduct.nunique()} distinct adducts; test uses {test.adduct.nunique()}")
in_test = train.adduct.isin(TEST_ADDUCTS)
print(f"train spectra whose adduct also occurs in test: {in_test.sum():,} ({in_test.mean():.1%})")
print("\\nionization mode:")
print(pd.DataFrame({"test": test.ionization_mode.value_counts(normalize=True),
                    "train": train.ionization_mode.value_counts(normalize=True)}).to_string(float_format="%.1%%"))
""")

md("""
## 6. Precursor mass — is train the same chemical size range as test?

The precursor m/z strongly constrains the molecular formula, so it is the backbone of any candidate
retrieval step. If train and test occupy different mass ranges, retrieval priors learned on train
will be miscalibrated.
""")

code("""
print("precursor_mz")
display(pd.DataFrame({"test": test.precursor_mz.describe(percentiles=[.05,.5,.95]),
                      "train": train.precursor_mz.describe(percentiles=[.05,.5,.95])}).round(1))

if PLOT:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    bins = np.linspace(100, 1300, 120)
    ax.hist(train.precursor_mz.dropna(), bins=bins, density=True, alpha=.45, label="train (all)")
    for L in ["enveda-180", "riken", "gnps"]:
        ax.hist(train.loc[train.ingest_lib == L, "precursor_mz"], bins=bins, density=True,
                histtype="step", lw=1.3, label=f"train · {L}")
    ax.hist(test.precursor_mz.dropna(), bins=bins, density=True, histtype="step",
            lw=2.2, color="k", label="TEST")
    ax.set_xlabel("precursor m/z"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Test sits where the natural-product libraries sit, not where enveda-180 sits")
    plt.tight_layout(); plt.show()
""")

md("""
## 7. Peak statistics and spectrum quality

How much signal is actually in a spectrum, and how aggressively should it be cleaned?
""")

code("""
%%time
# Peak arrays for a sample of train + all of test.
SAMPLE = 60_000
idx = np.sort(np.random.default_rng(0).choice(len(train), min(SAMPLE, len(train)), replace=False))
tr_peaks = pq.read_table(DATA / "train.parquet",
                         columns=PEAK_COLS + ["precursor_mz","ingest_lib"]).to_pandas().iloc[idx]
te_peaks = pq.read_table(DATA / "test.parquet", columns=PEAK_COLS + ["precursor_mz"]).to_pandas()
print(f"sampled {len(tr_peaks):,} train spectra, {len(te_peaks):,} test spectra")
""")

code("""
def peak_stats(df):
    n  = df.ms2_mzs.map(len)
    # how many peaks survive a 1% relative-intensity floor
    n1 = df.ms2_normalized_intensities.map(lambda a: int((np.asarray(a) >= 0.01).sum()))
    above = [float((np.asarray(m) > p + 1.5).mean()) if len(m) else 0.0
             for m, p in zip(df.ms2_mzs, df.precursor_mz)]
    return pd.DataFrame({"n_peaks": n, "n_peaks_1pct": n1, "frac_above_precursor": above})

ts, tes = peak_stats(tr_peaks), peak_stats(te_peaks)
display(pd.concat({"train(sample)": ts.describe(percentiles=[.5,.9,.99]),
                   "test": tes.describe(percentiles=[.5,.9,.99])}, axis=1).round(2))

print(f"\\ntest spectra with <6 peaks : {(tes.n_peaks < 6).mean():.1%}  "
      f"(a common pipeline would drop these — but you must still submit a row for them)")
print(f"test spectra with >128 peaks: {(tes.n_peaks > 128).mean():.1%}")
print(f"\\npeaks above precursor+1.5 Da — test: {tes.frac_above_precursor.mean():.3%} "
      f"(pre-cleaned by the host), train: {ts.frac_above_precursor.mean():.3%}")
""")

code("""
if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    bins = np.logspace(0, 3.6, 60)
    axes[0].hist(ts.n_peaks, bins=bins, density=True, alpha=.5, label="train")
    axes[0].hist(tes.n_peaks, bins=bins, density=True, histtype="step", lw=2, color="k", label="test")
    axes[0].set_xscale("log"); axes[0].set_xlabel("peaks per spectrum"); axes[0].legend()
    axes[0].axvline(6, color="crimson", ls=":"); axes[0].axvline(128, color="crimson", ls=":")
    axes[0].set_title("peak count (dotted: common 6 / 128 cutoffs)", fontsize=9)

    allint = np.concatenate(te_peaks.ms2_normalized_intensities.values[:3000])
    axes[1].hist(np.log10(allint + 1e-6), bins=80, density=True, color="k")
    for t in [0.001, 0.01, 0.02]:
        axes[1].axvline(np.log10(t), ls=":", color="crimson")
    axes[1].set_xlabel("log10 relative intensity"); axes[1].set_title(
        "test peak intensities (dotted: 0.1% / 1% / 2% floors)", fontsize=9)
    plt.tight_layout(); plt.show()

    frac = [float((np.asarray(a) >= t).mean()) for t in [0.001,0.01,0.02,0.05]
            for a in te_peaks.ms2_normalized_intensities.values[:4000]]
    frac = np.array(frac).reshape(4, -1).mean(1)
    print("mean fraction of test peaks kept by floor:",
          dict(zip(["0.1%","1%","2%","5%"], np.round(frac, 3))))
""")

md("""
## 8. Collision energy — three columns because the sources disagree

`collision_energy_orig` is lossless but unusable (free text, mixed units).
`collision_energy_ev` is the host's conversion to eV and **is the column that lines up with test**.
""")

code("""
print("collision_energy_orig_units in train:")
display(train.collision_energy_orig_units.value_counts(dropna=False).rename("spectra").to_frame())
print(f"\\ncollision_energy_ev null in train: {train.collision_energy_ev.isna().mean():.1%}")
print("test collision_energy_orig_units:", test.collision_energy_orig_units.unique())

def ce_mean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return np.nan
    a = np.asarray(v, dtype=float).ravel()
    return a.mean() if a.size else np.nan

test["ce_mean"]  = test.collision_energy_ev.map(ce_mean)
test["ce_n"]     = test.collision_energy_ev.map(lambda v: 0 if v is None else len(np.atleast_1d(v)))
print(f"\\ntest: {(test.ce_n > 1).mean():.1%} of spectra are MERGED across several energies")
display(test.ce_mean.describe(percentiles=[.05,.5,.95]).round(1).to_frame("test ce (eV)").T)
""")

md("""
## 9. The hard part: how many molecules share a mass?

Candidate retrieval starts from the precursor mass. The number of *distinct structures* consistent
with that mass is the size of the haystack — and it is the reason this problem is hard.
""")

code("""
# Structures per molecular formula: a lower bound on the isomer burden, since
# real retrieval searches all of PubChem, not just this training set.
by_formula = train.groupby("molecular_formula").inchikey14.nunique()
print("distinct structures sharing one molecular formula (within train only):")
print(by_formula.describe(percentiles=[.5,.9,.99]).round(1).to_string())
print(f"\\nformulas with >1 structure: {(by_formula>1).mean():.1%}")
display(by_formula.sort_values(ascending=False).head(8).rename("structures in train").to_frame())
""")

code("""
# How many train structures fall inside a 10 ppm window around each test precursor?
struct = (train.drop_duplicates("inchikey14")[["inchikey14","normalized_smiles","molecular_formula"]]
          .reset_index(drop=True))

EL = {"C":12.0,"H":1.0078250319,"N":14.0030740052,"O":15.9949146221,"S":31.97207069,
      "P":30.97376151,"F":18.000938,"Cl":34.96885271,"Br":78.9183376,"I":126.904468,
      "Si":27.9769265,"Na":22.98976928,"K":38.96370649,"Se":79.9165218,"B":11.0093055}
import re
_re = re.compile(r"([A-Z][a-z]?)(\\d*)")
def fmass(f):
    if not isinstance(f, str): return np.nan
    return sum(EL.get(s,0.0)*(int(n) if n else 1) for s,n in _re.findall(f))

struct["mass"] = struct.molecular_formula.map(fmass)
struct = struct[struct.mass.between(50, 2000)].sort_values("mass").reset_index(drop=True)
masses = struct.mass.to_numpy()
print(f"{len(struct):,} unique train structures with a usable formula mass")

SHIFT = {"[M+H]+":1.007276,"[M+NH4]+":18.033825,"[M-H2O+H]+":-17.003289,
         "[M-2H2O+H]+":-35.013854,"[M+Na]+":22.989221,"[M+K]+":38.963158,
         "[M-H]-":-1.007276,"[M-H2O-H]-":-19.017841,"[M+CH2O2-H]-":44.998203,
         "[M+Cl]-":34.969401}
tm = test.assign(neutral=test.precursor_mz - test.adduct.map(SHIFT)).dropna(subset=["neutral"])
tm = tm.groupby("molecule_id").neutral.median()
print(f"neutral mass recovered for {len(tm)} / {test.molecule_id.nunique()} test molecules")

for ppm in [5, 10, 20]:
    tol = tm.to_numpy()[:, None] * ppm * 1e-6
    lo = np.searchsorted(masses, tm.to_numpy() - tol.ravel())
    hi = np.searchsorted(masses, tm.to_numpy() + tol.ravel())
    n = hi - lo
    print(f"{ppm:>3} ppm -> train structures per test molecule: "
          f"median={np.median(n):.0f}  mean={n.mean():.0f}  max={n.max()}  zero-hit={(n==0).mean():.1%}")
""")

md("""
Two readings of that table, and they pull in opposite directions:

- **`zero-hit`** is the fraction of test molecules for which the *entire training set* offers no
  structure at the right mass. Those are class 2 or class 3 — unreachable without an external
  structure database or a generative model.
- **`median`** is how many decoys the ranker must sort through for the rest. And this is the
  optimistic number: searching PubChem instead of train inflates it by orders of magnitude.

Note the public `test.parquet` is a sample *drawn from train*, so these coverage numbers are far
rosier than the hidden test set will be. Treat them as an upper bound, not an estimate.
""")

md("""
## 10. The metric: what MRR@25 actually rewards

$$\\text{MRR@25} = \\frac{1}{U}\\sum_u \\frac{1}{\\text{rank}_u}$$

Rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33, rank 10 → 0.1, rank 25 → 0.04, not in list → 0.
""")

code("""
r = np.arange(1, 26)
display(pd.DataFrame({"rank": r, "credit": (1/r).round(3)}).set_index("rank").T)

print("Marginal value of moving the correct answer up one place:")
for a, b in [(1,2),(2,3),(5,6),(10,11),(24,25)]:
    print(f"  {b} -> {a}: +{1/a - 1/b:.3f}")
for line in [
    "Reading:",
    "  * Moving a hit from rank 3 to rank 1 is worth more (+0.67) than finding 16 brand-new",
    "    answers at rank 25 (+0.04 each). Reranking a good candidate list beats widening it.",
    "  * But a molecule with NO correct candidate scores 0 however good the ranker is.",
    "    So: maximise candidate recall first, then spend everything on ranking.",
    "  * Filling all 25 slots is free - no penalty for a wrong guess beyond the slot it takes.",
    "    Never submit a short list.",
]:
    print(line)
""")

code("""
# What would a "perfect ranker over train structures" score? An upper bound for pure retrieval.
lookup = set(train.inchikey14.dropna())
print(f"train covers {len(lookup):,} distinct structures")
print("\\nIf the hidden test were split 40/40/20 across classes 1/2/3, and you ranked")
print("perfectly whenever the structure is reachable:")
for c1, c2, c3 in [(.4,.4,.2), (.5,.3,.2), (.34,.33,.33)]:
    print(f"  mix {c1:.0%}/{c2:.0%}/{c3:.0%}:  library-search-only ceiling = {c1:.2f}   "
          f"+ perfect DB retrieval = {c1+c2:.2f}   + de novo = {c1+c2+c3:.2f}")
print("\\nThat is the whole competition in one line: each class you can't touch is a hard cap.")
""")

md("## 11. The submission format")

code("""
display(sub.head(3))
print(f"\\nrows: {len(sub)}  == unique test molecules: {test.molecule_id.nunique()}")
n_guesses = sub.smiles.str.count(";") + 1
print(f"guesses per row in sample_submission: min={n_guesses.min()} max={n_guesses.max()}")

for line in [
    "Rejection rules (a rejected submission costs a full 9-hour run):",
    "  - must have exactly the columns molecule_id, smiles",
    "  - every molecule_id exactly once, none missing, none extra",
    "  - no nulls in either column",
    "  - at most 25 semicolon-separated guesses per row",
    "  - file must be named submission.csv; notebook runs offline in <= 9h",
]:
    print(line)
""")

code("""
# A syntactically valid (and useless) submission, to prove the format end to end.
demo = pd.DataFrame({
    "molecule_id": sorted(test.molecule_id.unique()),
    "smiles": ";".join(["CC(=O)Nc1ccc(O)cc1", "NCCc1ccc(O)cc1", "OC(=O)c1ccccc1O"]),
})
assert demo.molecule_id.is_unique and demo.notna().all().all()
assert (demo.smiles.str.count(";") + 1 <= 25).all()
demo.to_csv("submission.csv", index=False)
print("wrote submission.csv"); display(demo.head(3))
""")

md("""
## 12. Takeaways

**About the data**
1. `inchikey14` is the unit of truth. Dedup, evaluate and think in it — never in raw SMILES.
   Stereochemistry and tautomers are free.
2. Train is 11 libraries pulling in different directions. `enveda-180` is ~46% of the rows, is on the
   right instrument, and is the *wrong chemistry*. The natural-product libraries are the right
   chemistry on the wrong instruments. `enveda-np-examples` (1,151 spectra) is the only place both
   line up — it is the calibration set, not a training set.
3. Test is one instrument (timsTOF), ten adducts, and ~3 spectra per molecule that often differ in
   adduct and collision energy. Merging those spectra per molecule is a modelling decision, not
   plumbing.
4. Use `collision_energy_ev`; the other two CE columns are provenance.

**About the task**
5. It is retrieval + ranking, not classification. Candidate recall is a hard ceiling on MRR — measure
   the two stages separately, always.
6. The three novelty classes need three different mechanisms. A method that only does library search
   is capped at whatever fraction class 1 turns out to be, and that fraction is hidden.
7. Moving a hit from rank 3 to rank 1 is worth more than 16 extra guesses. Rerank before you widen.
8. Always fill all 25 slots — unused slots are free expected score thrown away.

**About validation**
9. A random spectrum-level split leaves sibling spectra of the same molecule in train, so it silently
   measures class 1 only and will read far too high. Split by molecule, assign simulated novelty
   classes, and block class-3 structures from the candidate database.
10. The public `test.parquet` is drawn from train and will be replaced at rerun. Any coverage number
    computed against it is an upper bound.

Next: `notebooks/02_baseline_submission.ipynb` turns this into a first scored submission.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(cells)} cells)")
