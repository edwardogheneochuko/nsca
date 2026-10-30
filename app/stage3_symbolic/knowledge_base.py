"""Stage 3 – disease knowledge base in HPO term space.

Each KNOWN disease gets a profile: how often each grounded term occurs among its
training patients, its core terms (>= 50 %), and its excluding terms (never seen
with the disease although characteristic of another disease).  Held-out diseases
are never in the knowledge base, which is what makes the new-disease test fair.

Provenance: profiles are learned from the training patients, not taken from a
clinical source.  The file is plain JSON, so a curated clinical profile can replace
it later without code changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CORE = 0.5
TYPICAL = 0.1
EXCLUDE_IF_OTHER_DISEASE_AT = 0.3
MIN_PATIENTS_FOR_EXCLUSION = 30


class TermSpace:
    """Maps dataset columns to grounded terms (HPO id or LOCAL:feature)."""

    def __init__(self, grounding: pd.DataFrame, is_antecedent: np.ndarray):
        self.grounding = grounding.reset_index(drop=True)
        sym = ~np.asarray(is_antecedent, bool)
        self.terms = sorted(set(self.grounding.loc[sym, "term"]))
        self.index = {t: i for i, t in enumerate(self.terms)}
        self.feature_term = [self.index.get(t, -1) if s else -1 for t, s in zip(self.grounding.term, sym)]
        self.M = np.zeros((len(self.feature_term), len(self.terms)), np.uint8)
        for f, t in enumerate(self.feature_term):
            if t >= 0:
                self.M[f, t] = 1
        self.antecedent_idx = np.flatnonzero(~sym)

    def to_terms(self, X: np.ndarray) -> np.ndarray:
        return ((X.astype(np.int32) @ self.M) > 0).astype(np.uint8)


class KnowledgeBase:
    def __init__(self, profiles: dict, terms: list[str], term_labels: dict, provenance: str):
        self.profiles, self.terms, self.term_labels, self.provenance = profiles, terms, term_labels, provenance
        self.index = {t: i for i, t in enumerate(terms)}
        self.diseases = sorted(profiles)
        self.F = np.array([[profiles[d]["freq"].get(t, 0.0) for t in terms] for d in self.diseases], np.float32)
        self.core = self.F >= CORE
        self.excl = np.zeros_like(self.core)
        for i, d in enumerate(self.diseases):
            for t in profiles[d]["excluding"]:
                self.excl[i, self.index[t]] = True

    @classmethod
    def build(cls, T: np.ndarray, y: np.ndarray, space: TermSpace, X: np.ndarray, hpo, feature_meanings) -> "KnowledgeBase":
        profiles = {}
        diseases = sorted(set(y.tolist()))
        F = np.array([T[y == d].mean(0) for d in diseases])
        for i, d in enumerate(diseases):
            n = int((y == d).sum())
            freq = {space.terms[j]: round(float(F[i, j]), 4) for j in np.flatnonzero(F[i] > 0)}
            others = np.delete(F, i, axis=0)
            excl = []
            if n >= MIN_PATIENTS_FOR_EXCLUSION:
                excl = [space.terms[j] for j in np.flatnonzero((F[i] == 0) & (others.max(0) >= EXCLUDE_IF_OTHER_DISEASE_AT))]
            ante = {}
            if len(space.antecedent_idx):
                a = X[y == d][:, space.antecedent_idx].mean(0)
                ante = {feature_meanings[space.antecedent_idx[k]]: round(float(a[k]), 3) for k in np.flatnonzero(a >= 0.3)}
            profiles[d] = {"n_patients": n, "freq": freq,
                           "core": [t for t, f in freq.items() if f >= CORE],
                           "excluding": excl, "typical_antecedents": ante}
        labels = {t: (hpo.name(t) if t.startswith("HP:") else t.split(":", 1)[1]) for t in space.terms}
        return cls(profiles, space.terms, labels,
                   "Profiles learned from training patients of known diseases (not a clinical source).")

    def label(self, t: str) -> str:
        return self.term_labels.get(t, t)

    def save(self, path: Path):
        Path(path).write_text(json.dumps({"provenance": self.provenance, "terms": self.terms,
                                          "term_labels": self.term_labels, "diseases": self.profiles}, indent=1))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        j = json.loads(Path(path).read_text())
        return cls(j["diseases"], j["terms"], j["term_labels"], j["provenance"])
