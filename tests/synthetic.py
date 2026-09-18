"""A tiny fake dataset with the real schema, so the framework can be exercised
end-to-end without downloading 2.5M spectra (or accepting the rules first).

Structures are opaque identifier strings, not real SMILES — enough for the
harness, which only ever compares structure keys. Fragments are drawn from a
per-structure "true" peak set plus noise, so spectral similarity is genuinely
informative and a working pipeline should score well above chance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIBS = ["enveda-180", "gnps", "riken", "enveda-np-examples", "massbank", "pluskal_ms2"]
ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M-H]-", "[M+Cl]-"]
SHIFT = {"[M+H]+": 1.00728, "[M+NH4]+": 18.03383, "[M+Na]+": 22.98922,
         "[M-H]-": -1.00728, "[M+Cl]-": 34.96940}


def make_synthetic(n_structures: int = 200, seed: int = 0,
                   spectra_per_structure=(1, 6), n_isomers: int = 4) -> pd.DataFrame:
    """`n_isomers` structures share each exact mass, as real chemistry does.

    Without isomers the precursor mass alone identifies the molecule, and every
    model scores a perfect 1.0 — which tests nothing. Isomers are what force a
    model to actually read the fragments.
    """
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    masses = rng.uniform(150, 900, size=int(np.ceil(n_structures / n_isomers)))
    for k in range(n_structures):
        mass = float(masses[k // n_isomers])
        n_frag = int(rng.integers(6, 25))
        true_frags = np.sort(rng.uniform(50, mass, n_frag))     # the structure's signature
        base_int = rng.uniform(0.1, 1.0, n_frag)
        lib = LIBS[k % len(LIBS)]
        for _ in range(int(rng.integers(*spectra_per_structure))):
            adduct = ADDUCTS[int(rng.integers(len(ADDUCTS)))]
            prec = mass + SHIFT[adduct]
            keep = rng.random(n_frag) > 0.25                     # partial fragmentation
            if not keep.any():
                keep[rng.integers(n_frag)] = True
            mzs = true_frags[keep] + rng.normal(0, 0.003, keep.sum())
            ints = np.clip(base_int[keep] * rng.uniform(0.6, 1.4, keep.sum()), 1e-3, None)
            n_noise = int(rng.integers(0, 6))
            if n_noise:
                mzs = np.concatenate([mzs, rng.uniform(50, prec, n_noise)])
                ints = np.concatenate([ints, rng.uniform(1e-3, 0.08, n_noise)])
            order = np.argsort(mzs)
            mzs, ints = mzs[order], ints[order] / ints.max()
            rows.append(dict(
                molecule_id=f"mol_{k:05d}", spectrum_id=f"spec_{sid:06d}",
                ms2_mzs=mzs, ms2_normalized_intensities=ints,
                base_peak_intensity=float(rng.uniform(2e3, 1e6)),
                adduct=adduct, ionization_mode="positive" if adduct.endswith("+") else "negative",
                instrument_type="timsTOF", precursor_mz=prec,
                collision_energy_ev=[float(rng.choice([20, 40, 60]))],
                collision_energy_orig="40", collision_energy_orig_units="eV",
                normalized_smiles=f"STRUCT{k:05d}", inchikey=f"KEY{k:05d}-XXXXXXXX-N",
                inchikey14=f"KEY{k:05d}", molecular_formula="C10H12N2O",
                ingest_lib=lib, adduct_orig=adduct,
                precursor_error_ppm=float(rng.normal(0, 3)), num_peaks=len(mzs),
            ))
            sid += 1
    return pd.DataFrame(rows)
