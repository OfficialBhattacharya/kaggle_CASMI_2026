# Enveda CASMI 2026 — the problem, distilled

Source: <https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra>

## Task

Given the MS/MS spectra of an unknown small molecule, output **up to 25 candidate SMILES,
ranked best-first**. Predictions are **per molecule**, not per spectrum.

## Metric — MRR@25

`MRR@25 = (1/U) * sum_u 1/rank_u`, where `rank_u` is the position of the first correct
structure. Not in the list → 0.

Correctness is **not** SMILES string equality. Both prediction and answer are passed through
RDKit tautomer canonicalisation (pinned `2026.03.3`) and reduced to the **first block of the
InChIKey** (`inchikey14`). So:

- stereochemistry is free — do not model it
- tautomers are free
- `inchikey14` is the unit for dedup, evaluation and candidate sets

Credit by rank: 1 → 1.00, 2 → 0.50, 3 → 0.33, 10 → 0.10, 25 → 0.04.

## Data

| File | Contents |
|---|---|
| `train.parquet` | ~2.5M spectra, ~275k unique structures, 18 columns, labelled |
| `test.parquet` | ~1,500 spectra of ~400 molecules, 12 columns, unlabelled |
| `sample_submission.csv` | format reference |

The public `test.parquet` is **sampled from train** and is replaced by the hidden test set at
rerun. Any coverage statistic computed against it is an upper bound.

Peak lists are two aligned arrays (`ms2_mzs`, `ms2_normalized_intensities`), intensities
normalised so the base peak is 1.0.

### Test set specifics
- all Bruker **timsTOF**, 1–16 spectra per molecule (median 3)
- monoisotopic mass 157–1,159 Da (median 348)
- ten adducts: `[M+H]+ [M+NH4]+ [M-H2O+H]+ [M-2H2O+H]+ [M+Na]+ [M+K]+ [M-H]- [M-H2O-H]- [M+CH2O2-H]- [M+Cl]-`
- host cleaning: dropped mass-inconsistent spectra, dropped base peak < 1,000 counts,
  removed peaks above precursor + 2 Da

### Novelty classes (mix is hidden for the whole competition)

| Class | Definition | Only mechanism that reaches it |
|---|---|---|
| 1 | public reference spectra exist | spectral library search |
| 2 | in PubChem/COCONUT, no public spectra | database retrieval + reranking |
| 3 | not in PubChem at all | de novo generation |

### Training libraries

| ingest_lib | spectra | structures | character |
|---|---:|---:|---|
| enveda-180 | 1,153,785 | 182,941 | **right instrument (timsTOF), wrong chemistry** (synthetic screening) |
| pluskal_ms2 | 527,581 | 46,821 | Orbitrap, NCE units, consistent protocol |
| riken | 347,171 | 15,892 | plant specialised metabolites |
| gnps | 220,849 | 45,750 | largest NP collection, most heterogeneous |
| massbank | 101,727 | 9,180 | curated, many instruments |
| mona | 92,416 | 11,681 | community-hosted |
| spectraverse | 50,933 | 9,631 | harmonised aggregation |
| msdial | 40,765 | 9,127 | multi-instrument |
| drug_plus | 2,545 | 2,539 | ~1 spectrum each, no CE metadata |
| enveda-np-examples | 1,151 | 250 | **right instrument AND right chemistry** — the calibration set |
| masaryk | 652 | 416 | RECETOX standards |

`instrument_type` is free text in train (dozens of strings, many null); only `enveda-180` and
`enveda-np-examples` are uniformly timsTOF.

### Collision energy
Use **`collision_energy_ev`** — it is the column converted to match the test convention.
`collision_energy_orig` / `_units` are lossless provenance (eV, NCE, V, unknown). NCE→eV uses the
nominal Thermo formula and is approximate.

## Submission rules

`submission.csv`, columns `molecule_id,smiles`, guesses joined by `;`.

Rejected if: missing either column, empty, nulls, repeated `molecule_id`, or >25 guesses in a row.

Code competition: notebook, ≤9h CPU or GPU, **internet disabled**, output named `submission.csv`.
Freely and publicly available external data and pre-trained models are allowed.

## Timeline

- 2026-09-14 start
- 2026-12-07 entry + team merger deadline
- 2026-12-14 final submission

Prizes $50,000 (1st $16,000 … 5th $6,000).

## Named prior art (from the host)

MIST-CF and SIRIUS (formula annotation), matchms (spectrum handling), GNPS suspect annotations,
MassIVE (unlabelled repository-scale spectra), GeMS/DreaMS (pretrained spectrum embeddings),
FragHub (harmonised library aggregation).
