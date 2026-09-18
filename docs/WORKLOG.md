# Worklog

Shared memory across devices. Append at the top. Keep it short — numbers and decisions, not prose.
`experiments/results/leaderboard.csv` holds the machine-readable version; this file holds the *why*.

---

## 2026-09-18 — Kaggle wired up, EDA notebook pushed

**Kaggle auth.** The `kaggle` pip package caps at 1.7.4.5 on Python 3.9 and predates KGAT_ tokens —
it hard-requires a legacy `kaggle.json` and raises at import. `scripts/kaggle_api.py` talks to the
REST API directly with the token as a plain bearer, which works.

**Token scopes are partial.** This token has `competitions.list`, `competitions.data` and
`kernels.push`, but **not `kernels.get`** — so run status cannot be read from here. `--status` falls
back to the kernel listing and prints the URL. Regenerate the token with kernel read permission if
status polling matters.

**Pushed:** `02_baseline_submission.ipynb` ->
<https://www.kaggle.com/code/digantabhattacharya/casmi26-baseline-submission> (version 1, private).
Benchmarked before pushing: modified cosine is 45 us/pair and cleaning 8 us/spectrum, so even a
worst case of 3,000 library rows per mass window is ~3 min of matching for the whole test set.
Compute is not the constraint — **memory is**: train.parquet is 3 GB on disk and the peak arrays
expand well beyond that in pandas. If the notebook dies, it will be on the load, not the search.

**Pushed:** `01_eda.ipynb` -> <https://www.kaggle.com/code/digantabhattacharya/casmi26-eda-data-task-and-metric>
(version 1, private, internet off). Confirmed private: an unauthenticated GET returns 404. Note the
`isPrivate` field in the `/kernels/list` response is *not* populated — it reads `false` for private
kernels, so do not trust it; test anonymously instead.

**Already in flight (pre-dates this repo):** a submission from 16:03 UTC today,
"Enveda CASMI 2026 - Fast Spectral Cosine Baseline | Version 1", still `pending` with no score.
Its notebook is `digantabhattacharya/enveda-casmi-2026-fast-spectral-cosine-baseline`.
`notebooks/02_baseline_submission.ipynb` in this repo covers similar ground (cosine/modified-cosine
library search) — **compare the two before pushing it**, rather than spending a submission on a
duplicate approach.

**Data confirmed:** train.parquet 3,033 MB, test.parquet 4.8 MB, sample_submission.csv 44 KB.
The 3 GB train file is the practical constraint on any local work.

---

## 2026-09-18 — repo set up

Framework built and smoke-tested on synthetic data (`python3 tests/test_smoke.py`).
No real data pulled yet — competition rules not accepted at the time of writing.

**Decisions taken**

- **`inchikey14` is the unit of truth everywhere.** Dedup, candidate sets and scoring all key on it.
  `chem.KeyResolver` seeds SMILES→key from the training columns so RDKit is not on the hot path.
- **Validation simulates novelty classes** (`validation.make_split`). A plain random spectrum split
  leaves sibling spectra of the same molecule in train and therefore measures class 1 only — it will
  read far too high. Class 2 removes all of a molecule's spectra; class 3 also blocks the structure
  from the candidate database.
- **Validation over-samples natural-product libraries** (`prefer_libs`, weight 4×). `enveda-180` is
  46% of train, is on the right instrument, and is the wrong chemistry. Uniform sampling validates on
  the wrong distribution.
- **Library search indexes on neutral mass, not precursor m/z.** An [M+Na]+ library spectrum sits
  ~22 Da from the [M+H]+ query; a ppm window on precursor m/z hides it. Fixing this moved class-1 MRR
  from 0.250 to 1.000 on synthetic data.
- **Candidate generation is separated from ranking**, and `ScoreReport.candidate_recall` is reported
  alongside MRR. Recall is the hard ceiling; without it you cannot tell "never proposed" from
  "proposed but ranked badly".

**Findings from the synthetic smoke test** (synthetic data — directional only, not real numbers)

- Equal-weight RRF fusion of `library_search` + `candidate_prior` scored **0.331 vs 0.393** for the
  prior alone. It kept library search's class-1 strength (1.000) and gained class-2 reach (0.280),
  but library search's bad class-2 ordering diluted the prior's good one. → **Weight fusion members
  per class, or gate them on whether a library hit exists.** Do not assume fusion is free.
- Adding `meta` + `peak_stats` to a binned featurizer *hurt* k-NN (0.200 → 0.147): raw mass
  dominates the cosine and washes out the peak signal. → Scale or drop mass-like features before
  putting them in a distance-based model.

**Known limitation of local class-2 scores.** The candidate universe is built from this dataset, which
is far smaller than PubChem, so mass-window retrieval sees far fewer decoys than reality. Class-2
numbers are optimistic until a realistic structure table is attached.

**Next**

1. Accept competition rules, download data, run `01_eda.ipynb` on Kaggle.
2. Run the baseline notebook → first LB score. Record it here next to local CV.
3. Attach a PubChem/COCONUT subset as a Kaggle Dataset — the biggest single expected gain (class 2 is
   currently unreachable).
4. Formula prediction to narrow candidates before ranking.

---

## Template for new entries

```
## YYYY-MM-DD — <what changed>

**Change:** one line.
**Local CV:** MRR=____ (c1=__ c2=__ c3=__), candidate recall=____, split seed __
**LB:** ____
**Verdict:** kept / reverted, and why.
```
