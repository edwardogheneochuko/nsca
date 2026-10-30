"""Symptom data (Stage 1 input box): load either dataset into one common structure.

Disease labels are kept beside the symptom matrix and are never shown to ESOINN.
They are used only to name nodes, build the Stage 3 knowledge base from known
diseases, and evaluate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import DATA_DIR, SEED


@dataclass
class Dataset:
    name: str
    features: list[str]                 # column ids
    meanings: list[str]                 # plain-English meaning of each column
    is_antecedent: np.ndarray           # bool per feature (risk factor, not a symptom)
    X_train: np.ndarray                 # uint8, known diseases only
    y_train: np.ndarray
    X_val: np.ndarray                   # known diseases, for calibration / fidelity
    y_val: np.ndarray
    X_test: np.ndarray                  # known + held-out diseases
    y_test: np.ndarray
    test_is_new: np.ndarray             # True when the test patient's disease was held out
    known_diseases: list[str]
    heldout_diseases: list[str]
    test_differential: list | None = None   # DDXPlus only: ranked differential per test patient
    stream_order: np.ndarray | None = None  # training order for online learning
    info: dict = field(default_factory=dict)

    @property
    def symptom_mask(self) -> np.ndarray:
        return ~self.is_antecedent


def _split_known(X, y, frac_val=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    val_idx = []
    for d in np.unique(y):
        idx = np.flatnonzero(y == d)
        rng.shuffle(idx)
        val_idx.extend(idx[: max(1, int(round(len(idx) * frac_val)))])
    val = np.zeros(len(y), bool)
    val[val_idx] = True
    return ~val, val


def load_mendeley() -> Dataset:
    """43-disease subset of the Kaggle/Mendeley 773-disease dataset (DOI 10.17632/2cxccsxydc.1)."""
    d = DATA_DIR / "mendeley"
    df = pd.read_csv(d / "esoinn_subset.csv")
    info = json.loads((d / "subset_info.json").read_text())
    feats = [c for c in df.columns if c not in ("diseases", "count", "split")]
    X = df[feats].to_numpy(np.uint8)
    y = df["diseases"].to_numpy(str)
    counts = df["count"].to_numpy(int)
    new = (df["split"] == "heldout_new").to_numpy()

    Xk, yk, ck = X[~new], y[~new], counts[~new]
    tr, va = _split_known(Xk, yk)
    # Online stream: each unique pattern repeated by how often it occurred in the raw data.
    rng = np.random.default_rng(SEED)
    stream = np.repeat(np.flatnonzero(tr), ck[tr])
    tr_pos = {g: i for i, g in enumerate(np.flatnonzero(tr))}
    stream = np.array([tr_pos[g] for g in stream])
    rng.shuffle(stream)

    X_test = np.vstack([Xk[va], X[new]])
    y_test = np.concatenate([yk[va], y[new]])
    is_new = np.concatenate([np.zeros(va.sum(), bool), np.ones(new.sum(), bool)])
    # Validation = half of the known held-back rows; test keeps the other half + all new.
    kv = np.flatnonzero(~is_new)
    rng.shuffle(kv)
    half = kv[: len(kv) // 2]
    vmask = np.zeros(len(y_test), bool)
    vmask[half] = True
    return Dataset(
        name="mendeley", features=feats, meanings=feats,
        is_antecedent=np.zeros(len(feats), bool),
        X_train=Xk[tr], y_train=yk[tr],
        X_val=X_test[vmask], y_val=y_test[vmask],
        X_test=X_test[~vmask], y_test=y_test[~vmask], test_is_new=is_new[~vmask],
        known_diseases=list(info["known_diseases"]), heldout_diseases=list(info["heldout_new_diseases"]),
        stream_order=stream, info=info,
    )


def load_ddxplus() -> Dataset:
    """DDXPlus English (figshare doi:10.6084/m9.figshare.22687585), pre-encoded to 0/1 columns."""
    d = DATA_DIR / "ddxplus"
    info = json.loads((d / "ddxplus_info.json").read_text())
    fmeta = pd.read_csv(d / "ddxplus_features.csv")
    feats = fmeta["feature"].tolist()
    tr = pd.read_csv(d / "ddxplus_train.csv")
    te = pd.read_csv(d / "ddxplus_test.csv")
    Xk = tr[feats].to_numpy(np.uint8)
    yk = tr["pathology"].to_numpy(str)
    trm, vam = _split_known(Xk, yk)
    rng = np.random.default_rng(SEED)
    stream = rng.permutation(trm.sum())
    return Dataset(
        name="ddxplus", features=feats, meanings=fmeta["meaning"].tolist(),
        is_antecedent=fmeta["is_antecedent"].to_numpy(bool),
        X_train=Xk[trm], y_train=yk[trm], X_val=Xk[vam], y_val=yk[vam],
        X_test=te[feats].to_numpy(np.uint8), y_test=te["pathology"].to_numpy(str),
        test_is_new=(te["split"] == "heldout_new").to_numpy(),
        known_diseases=list(info["known_pathologies"]), heldout_diseases=list(info["heldout_new_pathologies"]),
        test_differential=te["differential"].tolist(), stream_order=stream, info=info,
    )


def load_phenopackets() -> Dataset:
    """Real published case reports from phenopacket-store, already coded in HPO (see scripts/build_phenopackets.py)."""
    d = DATA_DIR / "phenopackets"
    meta = json.loads((d / "meta.json").read_text())
    X = np.load(d / "X.npz")["X"]
    y = np.array([c["disease"] for c in meta["cases"]])
    sp = np.array([c["split"] for c in meta["cases"]])
    held = set(meta["heldout_new_diseases"])
    te = sp == "test"
    rng = np.random.default_rng(SEED)
    return Dataset(
        name="phenopackets", features=meta["features"], meanings=meta["meanings"],
        is_antecedent=np.zeros(len(meta["features"]), bool),
        X_train=X[sp == "train"], y_train=y[sp == "train"], X_val=X[sp == "val"], y_val=y[sp == "val"],
        X_test=X[te], y_test=y[te], test_is_new=np.array([v in held for v in y[te]]),
        known_diseases=meta["known_diseases"], heldout_diseases=meta["heldout_new_diseases"],
        stream_order=rng.permutation(int((sp == "train").sum())),
        info={k: v for k, v in meta.items() if k not in ("cases", "features", "meanings")},
    )


def load(name: str) -> Dataset:
    return {"mendeley": load_mendeley, "ddxplus": load_ddxplus, "phenopackets": load_phenopackets}[name]()
