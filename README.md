# CASMI 2026 — molecule ID from mass spectra

Working repo for the [Enveda CASMI 2026 Kaggle competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra):
predict a molecule's 2D structure (SMILES) from its MS/MS spectra, scored by **MRR@25**.

Read [`docs/COMPETITION.md`](docs/COMPETITION.md) first — it is the problem distilled to one page.
[`docs/WORKLOG.md`](docs/WORKLOG.md) is the running record of what has been tried and what it scored.

## Quick start

```bash
pip install -r requirements.txt
python3 tests/test_smoke.py          # end-to-end check on synthetic data, needs no download
```

With the real data (after accepting the competition rules):

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data --unzip
export CASMI_DATA=$PWD/data
python3 scripts/run_experiment.py --list
```

## Layout

```
notebooks/01_eda.ipynb                 EDA — the problem, the data, the metric
notebooks/02_baseline_submission.ipynb first scored submission (self-contained, runs offline)
src/casmi/
  config.py        paths that resolve the same on Kaggle and locally
  chem.py          adduct arithmetic, formula mass, InChIKey14 resolution
  spectra.py       peak cleaning (CleanConfig) + cosine / modified cosine
  metrics.py       MRR@25 with per-novelty-class breakdown
  validation.py    splits that simulate the three novelty classes
  features/        FEATURE FRAMEWORK — registry of swappable representations
  candidates/      candidate generation (the ceiling on MRR)
  models/          MODEL FRAMEWORK — registry of rankers, plus rank fusion
  experiment.py    one runner + one results table for both frameworks
  submit.py        build + validate submission.csv
experiments/results/leaderboard.csv    every local experiment, committed
docs/COMPETITION.md                    the problem, distilled
docs/WORKLOG.md                        decisions and results, newest first
```

## The two frameworks

Both funnel through one runner on one split, so their scores are directly comparable.

**Features** — hold the model fixed, vary the representation:

```python
from casmi.experiment import run_feature_experiment
from casmi.features import FeatureUnion, get_featurizer

run_feature_experiment(train, FeatureUnion([
    get_featurizer("binned", bin_width=0.5),
    get_featurizer("neutral_loss"),
]))
```

**Models** — hold the features fixed, vary the ranker:

```python
from casmi.experiment import run_model_experiment
from casmi.models import get_model
from casmi.candidates import get_candidates

run_model_experiment(train,
    get_model("library_search", similarity="modified"),
    candidates=get_candidates("mass_window", ppm=10.0))
```

Both append a row to `experiments/results/leaderboard.csv`, keyed by a hash of the full config, so
re-running the same thing updates rather than duplicates. **Commit that file** — it is what makes
working from several devices coherent.

Register your own with `@register_featurizer("name")` / `@register_model("name")` and it becomes
available to the runner and to `scripts/run_experiment.py`.

## Why validation is not a random split

A random spectrum-level split leaves other spectra of the same molecule in train, so it measures
novelty class 1 only and reads far too high. `validation.make_split` instead holds out whole
molecules, assigns each a simulated class, and degrades the training view to match — including
blocking class-3 structures from the candidate database. Always read the per-class breakdown, not
just the headline MRR; the real class mix is hidden for the whole competition.

Candidate recall is reported next to MRR because it is the hard ceiling: a structure never proposed
can never be ranked.

## Working from another device

```bash
git clone <this repo> && cd casmi26
pip install -r requirements.txt
python3 tests/test_smoke.py
```

Data is never committed (see `.gitignore`) — re-download it with the Kaggle CLI.
Notebook outputs are stripped before commit so diffs stay reviewable
(`python3 scripts/strip_notebooks.py`); the notebooks are generated from
`scripts/build_*_notebook.py`, so edit the script and regenerate rather than editing JSON.
