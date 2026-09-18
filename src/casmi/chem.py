"""Chemistry helpers: adduct arithmetic, formulas, and InChIKey14 matching.

RDKit is available on Kaggle but is optional here — everything that does not
strictly need it (adduct masses, formula parsing) works without it, so the
framework can be smoke-tested on a bare Python install.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

try:  # pragma: no cover - depends on environment
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDKIT = True
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # pragma: no cover
    RDKIT = False

PROTON = 1.00727646677
ELECTRON = 0.00054857990

MONOISOTOPIC = {
    "H": 1.0078250319, "C": 12.0, "N": 14.0030740052, "O": 15.9949146221,
    "P": 30.97376151, "S": 31.97207069, "F": 18.0009380, "Cl": 34.96885271,
    "Br": 78.9183376, "I": 126.904468, "Si": 27.9769265327, "Na": 22.98976928,
    "K": 38.96370649, "Se": 79.9165218, "B": 11.0093055, "As": 74.9215942,
}

# m/z of the ion = neutral monoisotopic mass + offset.  Signs already folded in.
ADDUCT_MASS_SHIFT: dict[str, float] = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 17.02654910 + PROTON,
    "[M-H2O+H]+": -18.01056468 + PROTON,
    "[M-2H2O+H]+": -2 * 18.01056468 + PROTON,
    "[M+Na]+": 22.98976928 - ELECTRON,
    "[M+K]+": 38.96370649 - ELECTRON,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -18.01056468 - PROTON,
    "[M+CH2O2-H]-": 46.00547931 - PROTON,
    "[M+Cl]-": 34.96885271 + ELECTRON,
    # a few extras that are common in train even though they are not in test
    "[M+2H]2+": PROTON,          # charge 2 -> m/z shift is per-charge
    "[M+CH3OH+H]+": 32.02621475 + PROTON,
    "[M+CH3CN+H]+": 41.02654910 + PROTON,
}

ADDUCT_CHARGE: dict[str, int] = {a: (1 if a.endswith("+") else -1) for a in ADDUCT_MASS_SHIFT}
ADDUCT_CHARGE["[M+2H]2+"] = 2

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}. Ignores charge suffixes."""
    counts: dict[str, int] = {}
    for sym, num in _FORMULA_RE.findall(formula or ""):
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    return counts


def formula_mass(formula: str) -> float:
    """Neutral monoisotopic mass of a molecular formula."""
    return sum(MONOISOTOPIC.get(s, 0.0) * n for s, n in parse_formula(formula).items())


def neutral_mass_from_precursor(precursor_mz: float, adduct: str) -> float:
    """Invert the adduct shift: measured m/z -> implied neutral mass M.

    Returns NaN for adducts we do not know, so callers can filter rather than
    silently get a wrong mass.
    """
    shift = ADDUCT_MASS_SHIFT.get(adduct)
    if shift is None:
        return float("nan")
    charge = abs(ADDUCT_CHARGE.get(adduct, 1))
    return precursor_mz * charge - shift * charge


def precursor_from_neutral_mass(mass: float, adduct: str) -> float:
    shift = ADDUCT_MASS_SHIFT.get(adduct)
    if shift is None:
        return float("nan")
    charge = abs(ADDUCT_CHARGE.get(adduct, 1))
    return (mass + shift * charge) / charge


def ppm_window(mass: float, ppm: float) -> tuple[float, float]:
    d = mass * ppm * 1e-6
    return mass - d, mass + d


# --- structure identity ----------------------------------------------------
@lru_cache(maxsize=200_000)
def _tautomer_canonical(smiles: str) -> str | None:
    if not RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=200_000)
def inchikey14(smiles: str) -> str | None:
    """The competition's correctness key: tautomer-canonicalised InChIKey block 1.

    Mirrors the stated grading procedure. Returns None when RDKit is absent or
    the SMILES does not parse — callers must treat None as 'not a match'.
    """
    if not RDKIT or not smiles:
        return None
    canon = _tautomer_canonical(smiles)
    if canon is None:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def inchikey14_many(smiles_iter: Iterable[str]) -> list[str | None]:
    return [inchikey14(s) for s in smiles_iter]


@lru_cache(maxsize=200_000)
def mol_formula(smiles: str) -> str | None:
    if not RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return CalcMolFormula(mol)


@lru_cache(maxsize=200_000)
def exact_mass(smiles: str) -> float:
    if not RDKIT:
        return float("nan")
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.ExactMolWt(mol) if mol is not None else float("nan")


class KeyResolver:
    """SMILES -> InChIKey14, using a learned lookup before touching RDKit.

    train.parquet already ships `inchikey14` for every labelled structure, so
    re-deriving it with RDKit for 275k structures is both slow and redundant.
    Seeding this from the training frame makes key resolution near-free and
    keeps the framework usable where RDKit is not installed at all.

    Resolution order: learned map -> RDKit -> the SMILES string itself. The last
    fallback is stricter than the real metric (no tautomer merging), so a score
    obtained without RDKit is a lower bound on the real one.
    """

    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    def learn(self, smiles_iter: Iterable[str], keys_iter: Iterable[str]) -> "KeyResolver":
        for smi, key in zip(smiles_iter, keys_iter):
            if smi and key and smi not in self._map:
                self._map[str(smi)] = str(key)
        return self

    def __call__(self, smiles: str) -> str:
        if not smiles:
            return ""
        hit = self._map.get(smiles)
        if hit is not None:
            return hit
        computed = inchikey14(smiles)
        if computed:
            self._map[smiles] = computed
            return computed
        return smiles

    def __len__(self) -> int:
        return len(self._map)


#: Process-wide resolver. `run_experiment` seeds it; notebooks may seed it too.
KEY_RESOLVER = KeyResolver()
