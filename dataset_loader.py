"""Load, validate, clean and preprocess the disease-symptom dataset.

Input  : data/original/disease_symptoms.csv   (never modified by this code)
Output : data/processed/                      (regenerated on every load)
           cases.csv            cleaned, de-duplicated cases (disease + pipe-separated symptoms)
           features.npz         X (binary matrix) and y_index (disease index per row)
           feature_names.json   the fixed symptom order and disease list
           manifest.json        sha256 of the original file, counts, preprocessing version

Expected original layout (the layout of the Kaggle "Disease Symptom Prediction" dataset;
verify its provenance and licence before use):

    Disease,Symptom_1,Symptom_2,...,Symptom_N

Cleaning steps: trim and collapse whitespace; normalise symptom names (lowercase, runs of
spaces/hyphens/underscores -> one underscore); drop rows with no disease or no symptoms;
drop exact duplicate cases (same disease AND same symptom set). Nothing is invented.
Symptom sets that appear under more than one disease are kept and counted, because they
are real ambiguity in the data.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PREPROCESSING_VERSION = "2"
DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ORIGINAL_PATH = DATA_DIR / "original" / "disease_symptoms.csv"
DEFAULT_PROCESSED_DIR = DATA_DIR / "processed"
DISEASE_COLUMN = "Disease"
SYMPTOM_COLUMN_PATTERN = re.compile(r"^symptom[_ ]?\d+$", re.IGNORECASE)
MIN_DISEASES = 2
MIN_ROWS = 10


class DatasetError(Exception):
    """Raised when the dataset is missing or invalid."""


@dataclass(frozen=True)
class LoadedDataset:
    original_path: str
    processed_dir: str
    sha256: str
    diseases: list[str]           # sorted unique disease names
    symptoms: list[str]           # fixed feature order (sorted normalised names)
    X: np.ndarray                 # (n_rows, n_symptoms) uint8 binary matrix
    y: np.ndarray                 # (n_rows,) disease names
    n_raw_rows: int
    n_rows_dropped: int           # missing disease or no symptoms
    n_duplicates_removed: int     # exact duplicate (disease, symptom set) cases
    n_conflicting_patterns: int   # identical symptom sets labelled with >1 disease
    processed_written: bool

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_diseases(self) -> int:
        return len(self.diseases)

    @property
    def n_symptoms(self) -> int:
        return len(self.symptoms)

    @property
    def symptom_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.symptoms)}


def normalise_symptom(raw: str) -> str:
    return re.sub(r"[\s_\-]+", "_", raw.strip().lower()).strip("_")


def vectorize(symptoms: Iterable[str], symptom_index: dict[str, int]) -> np.ndarray:
    """Binary feature vector in the dataset's fixed feature order. Unknown symptoms are ignored."""
    v = np.zeros(len(symptom_index), dtype=np.float64)
    for s in symptoms:
        i = symptom_index.get(normalise_symptom(s))
        if i is not None:
            v[i] = 1.0
    return v


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(
    original_path: str | Path | None = None,
    processed_dir: str | Path | None = None,
) -> LoadedDataset:
    p = Path(original_path) if original_path else DEFAULT_ORIGINAL_PATH
    out_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    if not p.exists():
        raise DatasetError(
            f"Dataset file not found at '{p}'. Place a verified, appropriately licensed "
            "disease-symptom CSV there (see README)."
        )
    try:
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise DatasetError(f"Could not read dataset '{p}': {exc}") from exc

    df.columns = [c.strip() for c in df.columns]
    if DISEASE_COLUMN not in df.columns:
        raise DatasetError(f"Dataset must have a '{DISEASE_COLUMN}' column; found {list(df.columns)}.")
    symptom_cols = [c for c in df.columns if SYMPTOM_COLUMN_PATTERN.match(c)]
    if not symptom_cols:
        raise DatasetError("Dataset has no symptom columns named like 'Symptom_1', 'Symptom_2', ...")

    n_raw = len(df)
    cases: list[tuple[str, frozenset[str]]] = []
    dropped = 0
    for disease, *cells in df[[DISEASE_COLUMN, *symptom_cols]].itertuples(index=False, name=None):
        disease = re.sub(r"\s+", " ", str(disease)).strip()
        syms = {normalise_symptom(c) for c in cells if isinstance(c, str) and c.strip()}
        syms.discard("")
        if not disease or not syms:
            dropped += 1
            continue
        cases.append((disease, frozenset(syms)))

    unique = list(dict.fromkeys(cases))  # keeps first-seen order
    n_dupes = len(cases) - len(unique)
    if len(unique) < MIN_ROWS:
        raise DatasetError(f"Dataset has only {len(unique)} usable unique cases (minimum {MIN_ROWS}).")

    diseases = sorted({d for d, _ in unique})
    if len(diseases) < MIN_DISEASES:
        raise DatasetError(f"Dataset has {len(diseases)} disease(s); at least {MIN_DISEASES} are required.")

    symptoms = sorted(set().union(*(s for _, s in unique)))
    index = {s: i for i, s in enumerate(symptoms)}
    X = np.zeros((len(unique), len(symptoms)), dtype=np.uint8)
    for r, (_, s) in enumerate(unique):
        for name in s:
            X[r, index[name]] = 1
    y = np.array([d for d, _ in unique])

    by_pattern: dict[frozenset[str], set[str]] = {}
    for d, s in unique:
        by_pattern.setdefault(s, set()).add(d)
    n_conflicts = sum(1 for ds in by_pattern.values() if len(ds) > 1)

    sha = _sha256(p)
    written = _write_processed(out_dir, unique, X, y, diseases, symptoms, sha,
                               n_raw, dropped, n_dupes, n_conflicts)
    return LoadedDataset(
        original_path=str(p), processed_dir=str(out_dir), sha256=sha, diseases=diseases,
        symptoms=symptoms, X=X, y=y, n_raw_rows=n_raw, n_rows_dropped=dropped,
        n_duplicates_removed=n_dupes, n_conflicting_patterns=n_conflicts, processed_written=written,
    )


def _write_processed(out_dir, unique, X, y, diseases, symptoms, sha, n_raw, dropped, dupes, conflicts) -> bool:
    """Best effort: a read-only filesystem must not stop the API from starting."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"disease": [d for d, _ in unique], "symptoms": ["|".join(sorted(s)) for _, s in unique]}
        ).to_csv(out_dir / "cases.csv", index=False)
        d_idx = {d: i for i, d in enumerate(diseases)}
        np.savez_compressed(out_dir / "features.npz", X=X, y_index=np.array([d_idx[d] for d in y]))
        (out_dir / "feature_names.json").write_text(json.dumps({"symptoms": symptoms, "diseases": diseases}, indent=2))
        (out_dir / "manifest.json").write_text(json.dumps({
            "preprocessing_version": PREPROCESSING_VERSION, "original_sha256": sha,
            "n_raw_rows": n_raw, "n_rows_dropped": dropped, "n_duplicates_removed": dupes,
            "n_training_rows": int(X.shape[0]), "n_diseases": len(diseases),
            "n_symptom_features": len(symptoms), "n_conflicting_patterns": conflicts,
        }, indent=2))
        return True
    except OSError:
        return False
