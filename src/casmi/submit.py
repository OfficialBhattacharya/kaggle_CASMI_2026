"""Build and validate submission.csv.

Validation mirrors every rejection rule stated in the competition's evaluation
section, because a rejected submission on a code competition costs a whole
run — cheaper to fail here than after a 9-hour notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .config import ID_COL, MAX_GUESSES, PATHS, SMILES_COL, SUBMISSION_NAME


class SubmissionError(ValueError):
    """Raised for anything Kaggle would reject."""


def build_submission(
    predictions: Mapping[str, Sequence[str]],
    molecule_ids: Iterable[str],
    fallback: Sequence[str] = ("C",),
) -> pd.DataFrame:
    """Assemble the two-column frame, one row per molecule_id.

    Every molecule_id must appear exactly once and nulls are rejected, so a
    molecule the model had nothing for gets `fallback` rather than an empty
    cell. A junk guess in slot 1 costs nothing that an empty row would not.
    """
    rows = []
    for mid in molecule_ids:
        guesses = [str(s) for s in predictions.get(mid, ()) if s and str(s).strip()]
        seen, uniq = set(), []
        for g in guesses:                     # duplicates waste a ranked slot
            if g not in seen:
                seen.add(g)
                uniq.append(g)
        if not uniq:
            uniq = list(fallback)
        rows.append({ID_COL: str(mid), SMILES_COL: ";".join(uniq[:MAX_GUESSES])})
    return pd.DataFrame(rows)


def validate_submission(df: pd.DataFrame, expected_ids: Iterable[str] | None = None) -> None:
    """Raise SubmissionError on anything the grader would reject."""
    missing = {ID_COL, SMILES_COL} - set(df.columns)
    if missing:
        raise SubmissionError(f"missing column(s): {sorted(missing)}")
    if df.empty:
        raise SubmissionError("submission is empty")
    if df[ID_COL].isna().any() or df[SMILES_COL].isna().any():
        raise SubmissionError("nulls present in molecule_id or smiles")
    if (df[SMILES_COL].astype(str).str.strip() == "").any():
        raise SubmissionError("blank smiles field present")
    dups = df[ID_COL][df[ID_COL].duplicated()].unique()
    if len(dups):
        raise SubmissionError(f"repeated molecule_id: {list(dups[:5])}")

    counts = df[SMILES_COL].astype(str).str.count(";") + 1
    if (counts > MAX_GUESSES).any():
        bad = df.loc[counts > MAX_GUESSES, ID_COL].head(5).tolist()
        raise SubmissionError(f">{MAX_GUESSES} guesses for molecule_id: {bad}")

    if expected_ids is not None:
        expected = set(map(str, expected_ids))
        got = set(df[ID_COL].astype(str))
        if expected - got:
            raise SubmissionError(f"{len(expected - got)} molecule_id(s) absent, "
                                  f"e.g. {sorted(expected - got)[:5]}")
        if got - expected:
            raise SubmissionError(f"{len(got - expected)} unexpected molecule_id(s), "
                                  f"e.g. {sorted(got - expected)[:5]}")


def check_smiles_parse(df: pd.DataFrame, sample: int = 500) -> dict:
    """Report how many guesses RDKit can actually parse.

    Not a rejection rule — an unparseable SMILES simply never matches — but a
    generative model quietly emitting invalid strings is the failure mode this
    catches, and it is invisible in the score until you look.
    """
    from .chem import RDKIT
    if not RDKIT:
        return {"rdkit": False}
    from rdkit import Chem
    total = bad = 0
    for cell in df[SMILES_COL].astype(str).head(sample):
        for smi in cell.split(";"):
            total += 1
            if Chem.MolFromSmiles(smi) is None:
                bad += 1
    return {"rdkit": True, "checked": total, "unparseable": bad,
            "frac_bad": (bad / total) if total else 0.0}


def write_submission(df: pd.DataFrame, path: Path | str | None = None,
                     expected_ids: Iterable[str] | None = None) -> Path:
    """Validate, then write. Defaults to the working dir Kaggle collects from."""
    validate_submission(df, expected_ids)
    path = Path(path) if path else (PATHS.work / SUBMISSION_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
