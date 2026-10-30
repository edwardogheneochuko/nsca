"""Stage 2 – Exemplars ("k-medoids per cluster").

For each cluster, k real training patients (medoids) are chosen so that each
cluster is represented by actual cases rather than an averaged prototype.
"""
from __future__ import annotations

import numpy as np


def jaccard_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1, keepdims=True) + B.sum(1)[None, :] - inter
    return 1.0 - inter / np.maximum(union, 1e-9)


def k_medoids(D: np.ndarray, k: int, iters: int = 30, seed: int = 42) -> np.ndarray:
    n = len(D)
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    med = [int(np.argmin(D.sum(1)))]             # most central point first
    while len(med) < k:                           # then k-medoids++ style spread
        dmin = D[:, med].min(1)
        p = dmin ** 2
        med.append(int(rng.choice(n, p=p / p.sum())) if p.sum() > 0 else int(rng.integers(n)))
    med = np.array(med)
    for _ in range(iters):
        assign = np.argmin(D[:, med], 1)
        new = med.copy()
        for c in range(k):
            mem = np.flatnonzero(assign == c)
            if len(mem):
                new[c] = mem[np.argmin(D[np.ix_(mem, mem)].sum(1))]
        if np.array_equal(new, med):
            break
        med = new
    return med


class Exemplars:
    def __init__(self, k: int = 3, max_per_cluster: int = 400, seed: int = 42):
        self.k, self.max_per_cluster, self.seed = k, max_per_cluster, seed
        self.by_cluster: dict[int, dict] = {}

    def fit(self, X: np.ndarray, clusters: np.ndarray, labels: np.ndarray | None = None):
        rng = np.random.default_rng(self.seed)
        for c in np.unique(clusters):
            idx = np.flatnonzero(clusters == c)
            if len(idx) > self.max_per_cluster:
                idx = rng.choice(idx, self.max_per_cluster, replace=False)
            D = jaccard_matrix(X[idx], X[idx])
            med = idx[k_medoids(D, self.k, seed=self.seed)]
            self.by_cluster[int(c)] = {"train_index": med.tolist(), "X": X[med].astype(np.uint8),
                                       "labels": labels[med].tolist() if labels is not None else None}
        return self

    def nearest(self, x: np.ndarray, cluster: int, meanings: list[str], n: int = 2) -> list[dict]:
        e = self.by_cluster.get(int(cluster))
        if not e:
            return []
        d = jaccard_matrix(x.reshape(1, -1), e["X"])[0]
        out = []
        for j in np.argsort(d)[:n]:
            ex = e["X"][j]
            shared = np.flatnonzero((ex > 0) & (x > 0))
            out.append({"train_index": int(e["train_index"][j]), "distance": round(float(d[j]), 3),
                        "diagnosis": e["labels"][j] if e["labels"] else None,
                        "symptoms": [meanings[f] for f in np.flatnonzero(ex)],
                        "shared_with_patient": [meanings[f] for f in shared]})
        return out
