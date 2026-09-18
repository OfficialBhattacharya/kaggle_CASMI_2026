"""The experiment runner — one scoreboard for feature and model experiments.

Both frameworks the project is built around funnel through `run_experiment`, so
a featurizer change and a model change land in the same results table on the
same split and are directly comparable.

    run_feature_experiment(df, featurizer=...)   # model held fixed
    run_model_experiment(df, model=...)          # features held fixed

Results append to experiments/results/leaderboard.csv, keyed by a hash of the
full config, so re-running the same thing overwrites rather than duplicates.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .candidates.base import BlockedCandidates, CandidateSource
from .chem import KEY_RESOLVER
from .config import KEY_COL, LABEL_COL, MAX_GUESSES, PATHS
from .features.base import Featurizer, FeatureUnion, build_queries, get_featurizer
from .metrics import ScoreReport, score
from .models.base import PaddedRanker, Ranker, get_model
from .spectra import CleanConfig
from .validation import Split, make_split


@dataclass
class ExperimentConfig:
    """Everything that can change a score, in one hashable place."""
    name: str
    kind: str = "model"                       # "feature" | "model" | "full"
    clean: CleanConfig = field(default_factory=CleanConfig)
    featurizer: str = "binned"                # label only; object passed separately
    model: str = "library_search"
    candidates: str = "none"
    split_seed: int = 0
    n_molecules: int = 400
    notes: str = ""

    def hash(self) -> str:
        blob = json.dumps({**asdict(self), "clean": self.clean.key()}, sort_keys=True,
                          default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    report: ScoreReport
    seconds: float
    predictions: dict[str, list[str]] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {"hash": self.config.hash(), **asdict(self.config)}
        row["clean"] = self.config.clean.key()
        row.update(self.report.to_row())
        row["seconds"] = round(self.seconds, 1)
        row["run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        row["host"] = platform.node()
        return row


def _key_of(smiles: str) -> str:
    """InChIKey14 via the shared resolver. See `chem.KeyResolver`."""
    return KEY_RESOLVER(smiles)


def run_experiment(
    df: pd.DataFrame,
    *,
    config: ExperimentConfig,
    model: Ranker,
    featurizer: Featurizer | None = None,
    candidates: CandidateSource | None = None,
    split: Split | None = None,
    fallback: Ranker | None = None,
    candidate_universe: str = "full",
    log: bool = True,
    keep_predictions: bool = False,
) -> ExperimentResult:
    """Fit on the split's train view, rank the held-out molecules, score.

    Class-3 blocking is applied here rather than left to the caller — it is the
    single easiest thing to forget, and forgetting it silently inflates the
    headline number.
    """
    if LABEL_COL not in df.columns:
        raise KeyError(f"{LABEL_COL!r} missing — run_experiment needs labelled data")
    split = split or make_split(df, seed=config.split_seed, n_molecules=config.n_molecules)
    # Seed SMILES -> InChIKey14 from the dataset's own columns. This is a public
    # chemical mapping, not a label, so it leaks nothing across the split.
    if KEY_COL in df.columns:
        KEY_RESOLVER.learn(df[LABEL_COL], df[KEY_COL])
    t0 = time.time()

    train_df = df.iloc[split.train_idx]
    eval_df = df.iloc[split.eval_idx]

    if candidates is not None:
        # The candidate database stands in for PubChem/COCONUT, which contain
        # class-2 structures even though no spectra exist for them. Fitting it
        # on the train view only would make class 2 unreachable and report a
        # flat zero. Class 3 stays blocked, as it is absent from PubChem too.
        #
        # Caveat worth remembering when reading class-2 numbers: a DB built from
        # this dataset is far smaller than real PubChem, so mass-window
        # retrieval returns far fewer decoys and class-2 scores here are
        # optimistic. Attach a realistic structure table to get an honest read.
        universe = df if candidate_universe == "full" else train_df
        candidates = BlockedCandidates(candidates, split.blocked_keys, _key_of)
        candidates.fit(universe)
    else:
        universe = train_df

    model.fit(train_df, featurizer) if featurizer is not None else model.fit(train_df)
    # Every source the model owns, however deeply nested, must see the same
    # universe and the same class-3 blocklist as the one used for diagnostics.
    model.rebind_sources(
        lambda src: BlockedCandidates(src, split.blocked_keys, _key_of).fit(universe))
    if fallback is not None:
        fallback.fit(train_df)
        fallback.rebind_sources(
            lambda src: BlockedCandidates(src, split.blocked_keys, _key_of).fit(universe))
    ranker = PaddedRanker(model, fallback) if fallback is not None else model

    queries = build_queries(eval_df, config.clean)
    predictions, pools = {}, {}
    for q in queries:
        smiles = ranker.rank(q, MAX_GUESSES)
        # class-3 structures must be unreachable however they were produced
        smiles = [s for s in smiles if _key_of(s) not in split.blocked_keys]
        predictions[q.molecule_id] = [_key_of(s) for s in smiles]
        if candidates is not None:
            pools[q.molecule_id] = {_key_of(c) for c in candidates.propose(q, 5000)}

    report = score(predictions, split.truth, split.novelty_class,
                   pools if candidates is not None else None)
    result = ExperimentResult(config, report, time.time() - t0,
                              predictions if keep_predictions else {})
    if log:
        log_result(result)
        print(f"[{config.hash()}] {config.name}\n{report}")
    return result


def run_feature_experiment(df, featurizer, *, name=None, model_name="feature_knn",
                           model_kwargs=None, clean=None, **kw) -> ExperimentResult:
    """Sweep features with the model pinned. Score deltas are feature deltas."""
    clean = clean or CleanConfig()
    cfg = ExperimentConfig(name=name or f"feat:{featurizer.name}", kind="feature",
                           clean=clean, featurizer=featurizer.name, model=model_name)
    return run_experiment(df, config=cfg, model=get_model(model_name, **(model_kwargs or {})),
                          featurizer=featurizer, **kw)


def run_model_experiment(df, model, *, name=None, featurizer=None, clean=None,
                         candidates=None, **kw) -> ExperimentResult:
    """Sweep models with the features pinned. Score deltas are model deltas."""
    clean = clean or CleanConfig()
    cfg = ExperimentConfig(name=name or f"model:{model.name}", kind="model", clean=clean,
                           featurizer=featurizer.name if featurizer else "n/a",
                           model=model.name,
                           candidates=candidates.name if candidates else "none")
    return run_experiment(df, config=cfg, model=model, featurizer=featurizer,
                          candidates=candidates, **kw)


# --- results table ---------------------------------------------------------
LEADERBOARD = "leaderboard.csv"


def log_result(result: ExperimentResult, path: Path | None = None) -> Path:
    """Append (or replace) one row in the local leaderboard, keyed by config hash."""
    path = path or (PATHS.results / LEADERBOARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([result.to_row()])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old.get("hash", pd.Series(dtype=str)) != result.config.hash()]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)
    return path


def leaderboard(path: Path | None = None, sort_by: str = "mrr") -> pd.DataFrame:
    """Every local experiment so far, best first. Commit this file — it is the
    shared memory that makes working from several devices coherent."""
    path = path or (PATHS.results / LEADERBOARD)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).sort_values(sort_by, ascending=False).reset_index(drop=True)
