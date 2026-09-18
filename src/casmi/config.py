"""Paths and constants that resolve identically on Kaggle and on a laptop.

Import this first in every notebook so the same code runs in both places.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

COMP_SLUG = "enveda-CASMI26-molecule-id-mass-spectra"

_KAGGLE_INPUT = Path("/kaggle/input") / COMP_SLUG
_REPO_ROOT = Path(__file__).resolve().parents[2]


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    work: Path

    @property
    def train(self) -> Path:
        return self.data / "train.parquet"

    @property
    def test(self) -> Path:
        return self.data / "test.parquet"

    @property
    def sample_submission(self) -> Path:
        return self.data / "sample_submission.csv"

    @property
    def results(self) -> Path:
        return self.root / "experiments" / "results"

    @property
    def cache(self) -> Path:
        return self.work / "cache"


def get_paths() -> Paths:
    """Data dir from CASMI_DATA env var, else Kaggle input, else ./data."""
    env = os.environ.get("CASMI_DATA")
    if env:
        data = Path(env)
        work = Path(os.environ.get("CASMI_WORK", _REPO_ROOT / "work"))
    elif on_kaggle():
        data = _KAGGLE_INPUT
        work = Path("/kaggle/working")
    else:
        data = _REPO_ROOT / "data"
        work = _REPO_ROOT / "work"
    return Paths(root=_REPO_ROOT, data=data, work=work)


PATHS = get_paths()

# --- competition constants -------------------------------------------------
MAX_GUESSES = 25          # MRR@25
SUBMISSION_NAME = "submission.csv"
ID_COL = "molecule_id"
SMILES_COL = "smiles"
LABEL_COL = "normalized_smiles"
KEY_COL = "inchikey14"    # what correctness is actually judged on

# The ten adducts that appear in the test set. Train has many more.
TEST_ADDUCTS = (
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
)

# Libraries acquired on the same Bruker timsTOF as the test set.
TIMSTOF_LIBS = ("enveda-180", "enveda-np-examples")
# Libraries whose chemistry looks like the test set (natural products).
NATURAL_PRODUCT_LIBS = ("enveda-np-examples", "riken", "gnps", "spectraverse", "massbank")
