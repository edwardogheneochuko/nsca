"""NSCA engine: trains ESOINN on the loaded dataset and identifies symptom patterns.

ESOINN is unsupervised. Disease labels are attached afterwards, per ESOINN cluster
(subclass), from the training rows whose nearest prototype node belongs to it. A cluster therefore only "represents"
a disease to the extent the data supports it (cluster support and purity are reported).

Outcomes
  known_pattern             input lies within a prototype's similarity threshold AND that
                            prototype is supported by enough, sufficiently pure rows of one disease
  potentially_novel_pattern input lies outside every learned prototype's similarity threshold;
                            flagged for human review. This does NOT mean a new disease exists.
  inconclusive              too little evidence, or the matching prototype is weak or ambiguous

The engine never ranks diseases and never returns more than one disease.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import numpy as np

from dataset_loader import LoadedDataset, normalise_symptom, vectorize
from esoinn import ESOINN, ESOINNConfig

KNOWN = "known_pattern"
NOVEL = "potentially_novel_pattern"
INCONCLUSIVE = "inconclusive"


@dataclass
class EngineConfig:
    epochs: int = 5
    seed: int = 42
    min_symptoms: int = 2            # fewer recognised symptoms -> inconclusive
    association_level: str = "node"  # "node" (prototype) or "cluster" (ESOINN subclass)
    min_support: int = 3             # training rows a prototype/cluster needs to name a disease
    purity_threshold: float = 0.8    # share of a cluster's rows that must be one disease
    ambiguity_margin: float = 0.10   # runner-up cluster this close, different disease -> inconclusive
    typical_symptom_share: float = 0.5


@dataclass
class ClusterProfile:
    disease: str
    support: int
    purity: float
    typical_symptoms: list[str]


@dataclass
class Identification:
    outcome: str
    reason_code: str
    disease: Optional[str]
    observed_symptoms: list[str]
    unrecognised_symptoms: list[str]
    cluster_id: Optional[int]
    node_id: Optional[int]
    matched_symptoms: list[str] = field(default_factory=list)
    typical_symptoms: list[str] = field(default_factory=list)
    cluster_support: Optional[int] = None
    cluster_purity: Optional[float] = None
    distance: Optional[float] = None
    similarity_threshold: Optional[float] = None
    explanation: str = ""
    human_review_required: bool = False


class NSCAEngine:
    def __init__(self, dataset: LoadedDataset, config: EngineConfig | None = None,
                 esoinn_config: ESOINNConfig | None = None):
        self.dataset = dataset
        self.cfg = config or EngineConfig()
        self.esoinn_cfg = esoinn_config or ESOINNConfig()
        self.model = ESOINN(self.esoinn_cfg)
        self.profiles: dict[int, ClusterProfile] = {}
        self.thresholds: dict[int, float] = {}
        self.symptom_index = dataset.symptom_index
        self.evaluation: Optional[dict] = None
        self._train(dataset.X, dataset.y)

    # ---------- training ----------
    def _train(self, X: np.ndarray, y: np.ndarray) -> None:
        Xf = X.astype(np.float64)
        self.model.fit(Xf, epochs=self.cfg.epochs, seed=self.cfg.seed)
        by_key: dict[int, list[int]] = {}
        for r in range(len(Xf)):
            nid, _, _, _ = self.model.nearest_two(Xf[r])
            by_key.setdefault(self._key(nid), []).append(r)
        symptoms = self.dataset.symptoms
        for key, rows in by_key.items():
            counts = Counter(y[rows])
            disease, n_dom = counts.most_common(1)[0]
            disease = str(disease)
            dom_rows = [r for r in rows if y[r] == disease]
            share = Xf[dom_rows].mean(axis=0)
            typical = [symptoms[k] for k in np.where(share >= self.cfg.typical_symptom_share)[0]]
            self.profiles[key] = ClusterProfile(disease, len(rows), n_dom / len(rows), typical)
        for nid in self.model.nodes:
            k = self._key(nid)
            t = self.model.similarity_threshold(nid)
            self.thresholds[k] = t if self.cfg.association_level == "node" else max(self.thresholds.get(k, 0.0), t)

    def _key(self, nid: int) -> int:
        """Unit that carries a disease association: the prototype node, or its ESOINN cluster."""
        return nid if self.cfg.association_level == "node" else self.model.nodes[nid].label

    def status(self) -> dict:
        return {
            "esoinn_trained": True,
            "n_nodes": self.model.n_nodes,
            "n_clusters": self.model.n_clusters,
            "esoinn_config": asdict(self.esoinn_cfg),
            "engine_config": asdict(self.cfg),
            "evaluation": self.evaluation,
        }

    # ---------- identification ----------
    def identify(self, symptoms: Iterable[str]) -> Identification:
        names = list(dict.fromkeys(normalise_symptom(s) for s in symptoms if s and s.strip()))
        observed = [s for s in names if s in self.symptom_index]
        unknown = [s for s in names if s not in self.symptom_index]

        if len(observed) < self.cfg.min_symptoms:
            return Identification(
                INCONCLUSIVE, "insufficient_evidence", None, observed, unknown, None, None,
                explanation=(f"Only {len(observed)} symptom(s) were recognised against the dataset's "
                             f"features; at least {self.cfg.min_symptoms} are needed. No result is forced."),
            )

        x = vectorize(observed, self.symptom_index)
        n1, n2, d1, d2 = self.model.nearest_two(x)
        node = self.model.nodes[n1]
        t1 = self.thresholds[self._key(n1)]
        base = dict(observed_symptoms=observed, unrecognised_symptoms=unknown,
                    cluster_id=node.label, node_id=n1, distance=round(d1, 4),
                    similarity_threshold=round(t1, 4))

        if d1 > t1:
            return Identification(
                NOVEL, "outside_similarity_threshold", None, **base, human_review_required=True,
                explanation=("This symptom combination lies outside the similarity range of every pattern "
                             "learned from the dataset. It may be unfamiliar, incomplete, or simply atypical. "
                             "This is not evidence of a new disease; it is flagged for human review."),
            )

        prof = self.profiles.get(self._key(n1))
        if prof is None or prof.support < self.cfg.min_support:
            return Identification(
                INCONCLUSIVE, "weak_support", None, **base, cluster_support=prof.support if prof else 0,
                explanation=(f"The closest learned pattern is backed by too few dataset cases "
                             f"(minimum {self.cfg.min_support}) to associate it with a disease."),
            )
        if prof.purity < self.cfg.purity_threshold:
            return Identification(
                INCONCLUSIVE, "mixed_cluster", None, **base, cluster_support=prof.support,
                cluster_purity=round(prof.purity, 3),
                explanation=("The closest learned pattern is shared by several diseases in the dataset, so "
                             "no single disease is supported."),
            )
        p2 = self.profiles.get(self._key(n2)) if n2 is not None else None
        if p2 is not None and p2.disease != prof.disease and d2 <= d1 * (1 + self.cfg.ambiguity_margin):
            return Identification(
                INCONCLUSIVE, "ambiguous_match", None, **base, cluster_support=prof.support,
                cluster_purity=round(prof.purity, 3),
                explanation=("Learned patterns associated with different diseases fit this input almost "
                             "equally well, so no single disease is supported."),
            )

        matched = [s for s in observed if s in prof.typical_symptoms]
        return Identification(
            KNOWN, "supported_pattern", prof.disease, **base, matched_symptoms=matched,
            typical_symptoms=prof.typical_symptoms, cluster_support=prof.support,
            cluster_purity=round(prof.purity, 3),
            explanation=(f"The symptoms fall within a learned pattern (cluster {node.label}) whose "
                         f"{prof.support} supporting dataset cases are {prof.purity:.0%} '{prof.disease}'. "
                         "This is a dataset association, not a diagnosis."),
        )


def evaluate_holdout(dataset: LoadedDataset, config: EngineConfig | None = None,
                     esoinn_config: ESOINNConfig | None = None, test_fraction: float = 0.2,
                     seed: int = 7, partial_k: int | None = None) -> dict:
    """Train on a stratified split and score on the held-out rows (full symptom vectors).

    Internal sanity check on dataset-derived data only. It is not clinical validation and
    says nothing about partial or real-world symptom descriptions.
    """
    cfg = config or EngineConfig()
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for d in dataset.diseases:
        rows = rng.permutation(np.where(dataset.y == d)[0])
        n_test = int(len(rows) * test_fraction) if len(rows) >= 3 else 0
        test_idx += list(rows[:n_test])
        train_idx += list(rows[n_test:])
    if not test_idx:
        return {"performed": False, "note": "Too few cases per disease to hold out a test set."}

    sub = LoadedDataset(**{**dataset.__dict__, "X": dataset.X[train_idx], "y": dataset.y[train_idx]})
    engine = NSCAEngine(sub, cfg, esoinn_config)
    counts = Counter()
    correct = 0
    for r in test_idx:
        ks = np.where(dataset.X[r] == 1)[0]
        if partial_k is not None and len(ks) > partial_k:
            ks = rng.choice(ks, size=partial_k, replace=False)
        syms = [dataset.symptoms[k] for k in ks]
        res = engine.identify(syms)
        counts[res.outcome] += 1
        if res.outcome == KNOWN and res.disease == dataset.y[r]:
            correct += 1
    n_test = len(test_idx)
    n_known = counts[KNOWN]
    return {
        "performed": True, "n_train": len(train_idx), "n_test": n_test,
        "known_pattern": counts[KNOWN], "potentially_novel_pattern": counts[NOVEL],
        "inconclusive": counts[INCONCLUSIVE],
        "known_pattern_accuracy": round(correct / n_known, 4) if n_known else None,
        "coverage": round(n_known / n_test, 4),
        "note": "Held-out split of the same dataset; not clinical validation.",
    }