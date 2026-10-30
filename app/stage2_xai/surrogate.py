"""Stage 2 – CART surrogate ("refit on trigger").

A decision tree learns to predict ESOINN's cluster from the symptoms.  Fidelity is
its agreement with ESOINN on patients it was not fitted on.  The tree is refitted
when ESOINN's structure changes (nodes added or removed, clusters merged or split)
or when fidelity falls below a threshold.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

FIDELITY_THRESHOLD = 0.90


class CARTSurrogate:
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 3, seed: int = 42):
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                           random_state=seed)
        self.fidelity: float | None = None
        self.fitted_version: int | None = None
        self.fitted_clusters: frozenset | None = None
        self.n_refits = 0

    def fit(self, X, esoinn_clusters, X_hold, hold_clusters, esoinn_version: int):
        self.tree.fit(X, esoinn_clusters)
        self.fidelity = float((self.tree.predict(X_hold) == hold_clusters).mean())
        self.fitted_version = esoinn_version
        self.fitted_clusters = frozenset(np.unique(esoinn_clusters).tolist())
        self.n_refits += 1
        return self

    def refit_triggers(self, esoinn) -> list[str]:
        """Return the reasons a refit is due (empty list = tree is still valid)."""
        reasons = []
        if self.fitted_version != esoinn.structure_version:
            reasons.append("ESOINN structure changed (node added/removed or edge cut)")
        current = frozenset(np.unique(esoinn.cluster[esoinn.nodes]).tolist())
        if current != self.fitted_clusters:
            reasons.append("cluster merge or split")
        if self.fidelity is not None and self.fidelity < FIDELITY_THRESHOLD:
            reasons.append(f"fidelity {self.fidelity:.2f} below {FIDELITY_THRESHOLD}")
        return reasons

    def predict(self, X):
        return self.tree.predict(X)

    def save(self, path: Path):
        joblib.dump(self, Path(path) / "cart_surrogate.joblib")

    @staticmethod
    def load(path: Path) -> "CARTSurrogate":
        return joblib.load(Path(path) / "cart_surrogate.joblib")
