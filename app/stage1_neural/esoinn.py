"""Stage 1 – ESOINN (Enhanced Self-Organizing Incremental Neural Network).

Follows Shen & Hasegawa (2008), "An enhanced self-organizing incremental neural
network for online unsupervised learning", Neural Networks 20(8), with two
changes from the project design:

* Distance is soft Jaccard (Tanimoto) instead of Euclidean, because patients
  are sparse binary symptom vectors.  For a binary patient x and a real-valued
  prototype w in [0,1]:  d = 1 - sum(min(x,w)) / sum(max(x,w)).
* Nodes removed as noise are not forgotten: the patients they represented go
  to a noise buffer, so a small cluster that might be a new disease can still
  be examined by Stage 3.

The model never sees disease labels while learning.  `assign_labels` is run
afterwards and only names nodes for explanation and evaluation.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _to_py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


class ESOINN:
    def __init__(self, dim: int, lam: int = 300, age_max: int = 50,
                 c1: float = 0.001, c2: float = 1.0, t_max: float = 0.7, seed: int = 42,
                 max_ids_per_node: int = 50, outlier_quantile: float = 0.95):
        self.dim, self.lam, self.age_max, self.c1, self.c2 = dim, lam, age_max, c1, c2
        # Project change: similarity thresholds are capped at t_max (Jaccard distance), so that
        # a node with no neighbours cannot accept patients from an unrelated disease.
        self.t_max = t_max
        self.outlier_quantile = outlier_quantile
        self.rng = np.random.default_rng(seed)
        self.max_ids = max_ids_per_node
        cap = 256
        self.W = np.zeros((cap, dim), np.float32)
        self.rowsum = np.zeros(cap, np.float32)
        self.active = np.zeros(cap, bool)
        self.wins = np.zeros(cap, np.float64)          # M_i
        self.point_sum = np.zeros(cap, np.float64)     # density points this period
        self.point_total = np.zeros(cap, np.float64)   # accumulated over finished periods
        self.periods = np.zeros(cap, np.float64)       # periods in which node won
        self.subclass = -np.ones(cap, np.int64)        # -1 = not yet assigned (new node)
        self.edges: dict[int, dict[int, int]] = defaultdict(dict)
        self.sample_ids: dict[int, list[int]] = defaultdict(list)
        self.noise_buffer: list[dict] = []
        self.n_seen = 0
        self.structure_version = 0
        # filled by finalize()
        self.cluster = None
        self.threshold = None
        self.node_label = None
        self.node_label_purity = None
        self.cluster_label = None
        self.cluster_size = None
        self.cluster_radius = {}
        self.rejected_ids: list[int] = []

    # ------------------------------------------------------------------ basics
    def _grow(self):
        cap = len(self.active) * 2
        def g(a, fill=0):
            out = np.full((cap,) + a.shape[1:], fill, a.dtype)
            out[: len(a)] = a
            return out
        self.W, self.rowsum, self.active = g(self.W), g(self.rowsum), g(self.active)
        self.wins, self.point_sum = g(self.wins), g(self.point_sum)
        self.point_total, self.periods = g(self.point_total), g(self.periods)
        self.subclass = g(self.subclass, -1)

    def _new_node(self, x: np.ndarray, sid: int | None) -> int:
        free = np.flatnonzero(~self.active)
        if len(free) == 0:
            self._grow()
            free = np.flatnonzero(~self.active)
        i = int(free[0])
        self.W[i] = x
        self.rowsum[i] = x.sum()
        self.active[i] = True
        self.wins[i] = 1
        self.point_sum[i] = self.point_total[i] = self.periods[i] = 0
        self.subclass[i] = -1
        self.edges[i] = {}
        self.sample_ids[i] = [sid] if sid is not None else []
        self.structure_version += 1
        return i

    def _remove_node(self, i: int, reason: str):
        for j in list(self.edges[i]):
            self.edges[j].pop(i, None)
        self.edges.pop(i, None)
        if self.sample_ids.get(i):
            self.noise_buffer.append({"node": i, "reason": reason, "sample_ids": self.sample_ids[i],
                                      "prototype_terms": int((self.W[i] >= 0.5).sum())})
        self.sample_ids.pop(i, None)
        self.active[i] = False
        self.W[i] = 0
        self.rowsum[i] = 0
        self.structure_version += 1

    @property
    def nodes(self) -> np.ndarray:
        return np.flatnonzero(self.active)

    def dist_to_nodes(self, x: np.ndarray, nodes: np.ndarray | None = None) -> np.ndarray:
        """Soft Jaccard distance from binary x to prototypes (fast path for binary x)."""
        nodes = self.nodes if nodes is None else nodes
        idx = np.flatnonzero(x)
        inter = self.W[np.ix_(nodes, idx)].sum(1) if len(idx) else np.zeros(len(nodes))
        union = len(idx) + self.rowsum[nodes] - inter
        return 1.0 - inter / np.maximum(union, 1e-9)

    def _proto_dist(self, i: int, js) -> np.ndarray:
        js = np.asarray(js, int)
        a, B = self.W[i], self.W[js]
        mn = np.minimum(a, B).sum(1)
        mx = np.maximum(a, B).sum(1)
        return 1.0 - mn / np.maximum(mx, 1e-9)

    def _threshold(self, i: int) -> float:
        nb = list(self.edges[i])
        if nb:
            t = float(self._proto_dist(i, nb).max())
        else:
            others = self.nodes[self.nodes != i]
            t = float(self._proto_dist(i, others).min()) if len(others) else np.inf
        return min(t, self.t_max)

    def density(self, i=None):
        tot = self.point_total + self.point_sum
        per = np.maximum(self.periods + (self.point_sum > 0), 1)
        h = tot / per
        return h if i is None else h[i]

    # ------------------------------------------------------------ subclasses
    @staticmethod
    def _alpha(mean_h: float, max_h: float) -> float:
        if 2 * mean_h >= max_h:
            return 0.0
        if 3 * mean_h >= max_h:
            return 0.5
        return 1.0

    def _subclass_stats(self):
        h = self.density()
        stats = {}
        groups = defaultdict(list)
        for i in self.nodes:
            groups[int(self.subclass[i])].append(i)
        for s, mem in groups.items():
            hv = h[mem]
            stats[s] = (hv.mean(), hv.max())
        return stats, h

    def _merge_ok(self, a: int, b: int, stats, h) -> bool:
        sa, sb = int(self.subclass[a]), int(self.subclass[b])
        if sa == -1 or sb == -1 or sa == sb:
            return True
        ma, xa = stats[sa]
        mb, xb = stats[sb]
        m = min(h[a], h[b])
        return m > self._alpha(ma, xa) * xa or m > self._alpha(mb, xb) * xb

    def _separate_subclasses(self):
        """Label subclasses by density apexes, then cut edges between subclasses that should stay apart."""
        h = self.density()
        nodes = self.nodes
        parent = {}
        for i in nodes:
            nb = list(self.edges[i])
            best = max(nb, key=lambda j: h[j]) if nb else i
            parent[i] = best if nb and h[best] > h[i] else i
        def root(i):
            seen = []
            while parent[i] != i:
                seen.append(i)
                i = parent[i]
            for s in seen:
                parent[s] = i
            return i
        for i in nodes:
            self.subclass[i] = root(i)
        stats, h = self._subclass_stats()
        # merge subclasses whose boundary nodes are dense enough, otherwise remove the edge
        changed = True
        while changed:
            changed = False
            for i in nodes:
                for j in list(self.edges[i]):
                    if self.subclass[i] == self.subclass[j]:
                        continue
                    if self._merge_ok(i, j, stats, h):
                        old, new = self.subclass[j], self.subclass[i]
                        self.subclass[self.subclass == old] = new
                        stats, h = self._subclass_stats()
                        changed = True
                    else:
                        self.edges[i].pop(j, None)
                        self.edges[j].pop(i, None)
                        self.structure_version += 1

    def _denoise(self):
        h = self.density()
        nodes = self.nodes
        if len(nodes) < 3:
            return
        mean_h = h[nodes].mean()
        for i in nodes:
            k = len(self.edges[i])
            if (k == 2 and h[i] < self.c1 * mean_h) or (k == 1 and h[i] < self.c2 * mean_h) or k == 0:
                self._remove_node(i, f"low density, {k} neighbour(s)")

    def _end_period(self):
        self._separate_subclasses()
        self._denoise()
        won = self.point_sum > 0
        self.point_total[won] += self.point_sum[won]
        self.periods[won] += 1
        self.point_sum[:] = 0

    # ---------------------------------------------------------------- learning
    def partial_fit(self, x: np.ndarray, sid: int | None = None) -> dict:
        x = x.astype(np.float32)
        self.n_seen += 1
        nodes = self.nodes
        if len(nodes) < 2:
            self._new_node(x, sid)
            return {"event": "new_node"}
        d = self.dist_to_nodes(x, nodes)
        o = np.argsort(d)[:2]
        s1, s2 = int(nodes[o[0]]), int(nodes[o[1]])
        if d[o[0]] > self._threshold(s1) or d[o[1]] > self._threshold(s2):
            self._new_node(x, sid)
            event = "new_node"
        else:
            stats, h = self._subclass_stats()
            if s2 in self.edges[s1] or self._merge_ok(s1, s2, stats, h):
                if s2 not in self.edges[s1]:
                    self.structure_version += 1
                self.edges[s1][s2] = 0
                self.edges[s2][s1] = 0
            else:
                self.edges[s1].pop(s2, None)
                self.edges[s2].pop(s1, None)
            for j in list(self.edges[s1]):
                if j != s2:
                    self.edges[s1][j] += 1
                    self.edges[j][s1] = self.edges[s1][j]
            # density point: closer to its neighbours = denser
            nb = list(self.edges[s1])
            if nb:
                md = float(self._proto_dist(s1, nb).mean())
                self.point_sum[s1] += 1.0 / (1.0 + md) ** 2
            self.wins[s1] += 1
            self.W[s1] += (x - self.W[s1]) / self.wins[s1]
            self.rowsum[s1] = self.W[s1].sum()
            for j in nb:
                self.W[j] += (x - self.W[j]) / (100.0 * self.wins[s1])
                self.rowsum[j] = self.W[j].sum()
            for j in list(self.edges[s1]):
                if self.edges[s1][j] > self.age_max:
                    self.edges[s1].pop(j)
                    self.edges[j].pop(s1, None)
                    self.structure_version += 1
            if sid is not None and len(self.sample_ids[s1]) < self.max_ids:
                self.sample_ids[s1].append(sid)
            event = "update"
        if self.n_seen % self.lam == 0:
            self._end_period()
        return {"event": event}

    def fit(self, X: np.ndarray, order: np.ndarray | None = None, epochs: int = 1, log=None):
        order = np.arange(len(X)) if order is None else order
        for ep in range(epochs):
            o = order if ep == 0 else self.rng.permutation(order)
            for t, i in enumerate(o):
                self.partial_fit(X[i], int(i))
                if log and (t + 1) % 5000 == 0:
                    log(f"epoch {ep + 1} step {t + 1}/{len(o)} nodes={len(self.nodes)}")
        self._end_period()
        return self

    # ----------------------------------------------------------- after training
    def components(self) -> dict[int, int]:
        comp, cid = {}, 0
        for i in self.nodes:
            if i in comp:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                u = stack.pop()
                for v in self.edges[u]:
                    if v not in comp:
                        comp[v] = cid
                        stack.append(v)
            cid += 1
        return comp

    def finalize(self, X: np.ndarray, y: np.ndarray | None = None, min_cluster_wins: int = 1):
        """Freeze cluster ids, node thresholds and (optionally) post-hoc node/cluster names."""
        comp = self.components()
        nodes = self.nodes
        self.cluster = -np.ones(len(self.active), np.int64)
        for i in nodes:
            self.cluster[i] = comp[i]
        self.threshold = np.full(len(self.active), np.inf)
        for i in nodes:
            self.threshold[i] = self._threshold(i)
        # Inference outlier rule, calibrated on training patients: a patient is an outlier when its
        # distance to the winner exceeds the q-quantile of training distances inside that cluster.
        s1, _, d1, _ = self.winners(X)
        cl = self.cluster[s1]
        glob = float(np.quantile(d1, self.outlier_quantile))
        self.cluster_radius = {}
        for c in np.unique(self.cluster[nodes]):
            dc = d1[cl == c]
            self.cluster_radius[int(c)] = float(np.quantile(dc, self.outlier_quantile)) if len(dc) >= 20 else glob
        # Noise buffer, final form: patients whose node was pruned as noise during learning AND who
        # are still outside every cluster's radius in the final model.  Stage 3 can examine them.
        pruned = sorted({i for g in self.noise_buffer for i in g["sample_ids"]})
        if pruned:
            out = d1[pruned] > np.array([self.cluster_radius.get(int(c), np.inf) for c in cl[pruned]])
            self.rejected_ids = [int(i) for i, o in zip(pruned, out) if o]
        else:
            self.rejected_ids = []
        # keep the pruned-node log compact: only groups that still hold rejected patients
        rej = set(self.rejected_ids)
        self.pruned_count = len(pruned)
        self.noise_buffer = [g for g in self.noise_buffer if rej.intersection(g["sample_ids"])]
        if y is not None:
            self.assign_labels(X, y)
        return self

    def winners(self, X: np.ndarray, batch: int = 2048):
        nodes = self.nodes
        Wn = self.W[nodes].astype(np.float32)
        rs = self.rowsum[nodes]
        s1 = np.empty(len(X), int); s2 = np.empty(len(X), int)
        d1 = np.empty(len(X)); d2 = np.empty(len(X))
        for a in range(0, len(X), batch):
            xb = X[a:a + batch].astype(np.float32)
            inter = xb @ Wn.T
            union = xb.sum(1, keepdims=True) + rs[None, :] - inter
            D = 1 - inter / np.maximum(union, 1e-9)
            o = np.argsort(D, 1)[:, :2]
            r = np.arange(len(xb))
            s1[a:a + batch] = nodes[o[:, 0]]; s2[a:a + batch] = nodes[o[:, 1]]
            d1[a:a + batch] = D[r, o[:, 0]]; d2[a:a + batch] = D[r, o[:, 1]]
        return s1, s2, d1, d2

    def runner_up_other_cluster(self, x: np.ndarray, s1: int) -> tuple[int, float]:
        nodes = self.nodes
        d = self.dist_to_nodes(x, nodes)
        mask = self.cluster[nodes] != self.cluster[s1]
        if not mask.any():
            return -1, float("nan")
        k = int(np.argmin(np.where(mask, d, np.inf)))
        return int(nodes[k]), float(d[k])

    def assign_labels(self, X, y):
        s1, _, _, _ = self.winners(X)
        by_node = defaultdict(list)
        for n, lab in zip(s1, y):
            by_node[n].append(lab)
        self.node_label = {}
        self.node_label_purity = {}
        for n in self.nodes:
            c = Counter(by_node.get(n, []))
            if c:
                lab, k = c.most_common(1)[0]
                self.node_label[int(n)] = lab
                self.node_label_purity[int(n)] = k / sum(c.values())
        by_cl = defaultdict(list)
        for n, lab in zip(s1, y):
            by_cl[int(self.cluster[n])].append(lab)
        self.cluster_label = {c: Counter(v).most_common(1)[0][0] for c, v in by_cl.items()}
        self.cluster_size = {c: len(v) for c, v in by_cl.items()}

    def predict(self, X):
        """Winner node, cluster, distance and whether the patient falls outside the winner's threshold."""
        s1, s2, d1, d2 = self.winners(X)
        cl = self.cluster[s1]
        radius = np.array([self.cluster_radius.get(int(c), np.inf) for c in cl])
        outlier = d1 > radius
        return {"winner": s1, "second": s2, "dist": d1, "dist2": d2,
                "cluster": self.cluster[s1], "outlier": outlier}

    # ------------------------------------------------------------- persistence
    def summary(self) -> dict:
        nodes = self.nodes
        n_edges = sum(len(v) for v in self.edges.values()) // 2
        return {"nodes": int(len(nodes)), "edges": int(n_edges),
                "clusters": int(len(set(self.cluster[nodes].tolist()))) if self.cluster is not None else None,
                "samples_seen": int(self.n_seen), "noise_buffer_groups": len(self.noise_buffer),
                "pruned_during_learning": int(getattr(self, "pruned_count", 0)),
                "noise_buffer_patients": len(self.rejected_ids),
                "structure_version": int(self.structure_version),
                "params": {"lambda": self.lam, "age_max": self.age_max, "c1": self.c1, "c2": self.c2,
                           "t_max": self.t_max, "outlier_quantile": self.outlier_quantile}}

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        n = len(self.active)
        np.savez_compressed(path / "esoinn_arrays.npz", W=self.W, rowsum=self.rowsum, active=self.active,
                            wins=self.wins, point_sum=self.point_sum, point_total=self.point_total,
                            periods=self.periods, subclass=self.subclass,
                            cluster=self.cluster if self.cluster is not None else -np.ones(n, np.int64),
                            threshold=self.threshold if self.threshold is not None else np.full(n, np.inf))
        meta = {"dim": self.dim, "lam": self.lam, "age_max": self.age_max, "c1": self.c1, "c2": self.c2,
                "t_max": self.t_max, "outlier_quantile": self.outlier_quantile,
                "cluster_radius": {str(k): v for k, v in self.cluster_radius.items()},
                "n_seen": self.n_seen, "structure_version": self.structure_version,
                "edges": {str(i): {str(j): a for j, a in v.items()} for i, v in self.edges.items() if self.active[i]},
                "sample_ids": {str(i): v for i, v in self.sample_ids.items()},
                "noise_buffer": self.noise_buffer, "rejected_ids": self.rejected_ids,
                "pruned_count": int(getattr(self, "pruned_count", 0)),
                "node_label": {str(k): v for k, v in (self.node_label or {}).items()},
                "node_label_purity": {str(k): v for k, v in (self.node_label_purity or {}).items()},
                "cluster_label": {str(k): v for k, v in (self.cluster_label or {}).items()},
                "cluster_size": {str(k): v for k, v in (self.cluster_size or {}).items()}}
        (path / "esoinn_state.json").write_text(json.dumps(meta, default=_to_py))

    @classmethod
    def load(cls, path: Path) -> "ESOINN":
        path = Path(path)
        meta = json.loads((path / "esoinn_state.json").read_text())
        m = cls(meta["dim"], meta["lam"], meta["age_max"], meta["c1"], meta["c2"], meta["t_max"],
                outlier_quantile=meta["outlier_quantile"])
        m.cluster_radius = {int(k): v for k, v in meta["cluster_radius"].items()}
        a = np.load(path / "esoinn_arrays.npz")
        for k in ("W", "rowsum", "active", "wins", "point_sum", "point_total", "periods", "subclass", "cluster", "threshold"):
            setattr(m, k, a[k])
        m.n_seen, m.structure_version = meta["n_seen"], meta["structure_version"]
        m.edges = defaultdict(dict, {int(i): {int(j): x for j, x in v.items()} for i, v in meta["edges"].items()})
        m.sample_ids = defaultdict(list, {int(k): v for k, v in meta["sample_ids"].items()})
        m.noise_buffer = meta["noise_buffer"]
        m.rejected_ids = meta.get("rejected_ids", [])
        m.pruned_count = meta.get("pruned_count", 0)
        m.node_label = {int(k): v for k, v in meta["node_label"].items()}
        m.node_label_purity = {int(k): v for k, v in meta["node_label_purity"].items()}
        m.cluster_label = {int(k): v for k, v in meta["cluster_label"].items()}
        m.cluster_size = {int(k): v for k, v in meta["cluster_size"].items()}
        return m
