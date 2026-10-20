"""ESOINN: Enhanced Self-Organizing Incremental Neural Network.

Implementation of the algorithm described in Furao, Ogata & Hasegawa (2007),
"An enhanced self-organizing incremental neural network for online unsupervised
learning", Neural Networks 20(8). Written from the published description, NOT checked
against the authors' reference code or reproduced on their benchmarks.

Interpretive choices (verify against the paper before citing this as "faithful"):
  * Density h_i = s_i / (signals presented since node i was inserted), where s_i
    accumulates p = 1/(1 + mean distance to topological neighbours)^2 each time i wins.
  * A subclass pair is merged when the overlap density (max over connecting edges of
    min(h_i, h_j)) >= alpha * apex density for EITHER subclass, with alpha = 0 if
    2*mean >= apex, 0.5 if 3*mean >= apex, else 1.0.
  * Winners that fall outside both similarity thresholds insert a new node; otherwise
    edge creation, ageing, density update, weight adaptation follow the paper's order
    (ageing before edge reset).
  * Noise removal never reduces the network below two nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ESOINNConfig:
    lambda_: int = 100     # period (in signals) of subclass partition + noise removal
    age_max: int = 50      # maximum edge age
    c1: float = 0.001      # noise threshold, nodes with 1 neighbour
    c2: float = 1.0        # noise threshold, nodes with 2 neighbours


@dataclass
class Node:
    id: int
    w: np.ndarray
    birth: int
    wins: int = 1
    s: float = 0.0
    density: float = 0.0
    label: int = -1  # subclass id; -1 = not yet classified ("new node")


class ESOINN:
    def __init__(self, config: ESOINNConfig | None = None):
        self.cfg = config or ESOINNConfig()
        self.nodes: dict[int, Node] = {}
        self.adj: dict[int, set[int]] = {}
        self.edges: dict[tuple[int, int], int] = {}  # (i, j) with i < j -> age
        self._next_id = 0
        self.n_signals = 0
        self._sub_stats: dict[int, tuple[float, float]] = {}  # label -> (apex density, mean density)

    # ---------- public API ----------
    def fit(self, X: np.ndarray, epochs: int = 1, seed: int = 0) -> "ESOINN":
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        for _ in range(epochs):
            for i in rng.permutation(len(X)):
                self.partial_fit(X[i])
        self._periodic_update()
        return self

    def partial_fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.n_signals += 1
        if len(self.nodes) < 2:
            self._add_node(x)
        else:
            a1, a2, d1, d2 = self.nearest_two(x)
            if d1 > self.similarity_threshold(a1) or d2 > self.similarity_threshold(a2):
                self._add_node(x)
            else:
                self._age_edges(a1)
                self._connect_or_disconnect(a1, a2)
                self._add_density_point(a1)
                self._adapt(a1, x)
                self._remove_old_edges()
        if self.n_signals % self.cfg.lambda_ == 0:
            self._periodic_update()

    def nearest_two(self, x: np.ndarray):
        """Return (winner id, runner-up id, winner distance, runner-up distance)."""
        ids = list(self.nodes)
        W = np.stack([self.nodes[i].w for i in ids])
        d = np.linalg.norm(W - x, axis=1)
        order = np.argsort(d, kind="stable")
        if len(ids) == 1:
            return ids[0], None, float(d[0]), float("inf")
        return ids[order[0]], ids[order[1]], float(d[order[0]]), float(d[order[1]])

    def similarity_threshold(self, i: int) -> float:
        wi = self.nodes[i].w
        if self.adj[i]:
            return max(float(np.linalg.norm(wi - self.nodes[j].w)) for j in self.adj[i])
        others = [float(np.linalg.norm(wi - n.w)) for k, n in self.nodes.items() if k != i]
        return min(others) if others else float("inf")

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_clusters(self) -> int:
        return len({n.label for n in self.nodes.values() if n.label >= 0})

    # ---------- internals ----------
    @staticmethod
    def _key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    def _add_node(self, x: np.ndarray) -> None:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(id=nid, w=x.copy(), birth=self.n_signals)
        self.adj[nid] = set()

    def _remove_node(self, i: int) -> None:
        for j in list(self.adj[i]):
            self._remove_edge(i, j)
        del self.adj[i]
        del self.nodes[i]

    def _remove_edge(self, i: int, j: int) -> None:
        self.edges.pop(self._key(i, j), None)
        self.adj[i].discard(j)
        self.adj[j].discard(i)

    def _age_edges(self, a1: int) -> None:
        for j in self.adj[a1]:
            self.edges[self._key(a1, j)] += 1

    def _remove_old_edges(self) -> None:
        for (i, j), age in list(self.edges.items()):
            if age > self.cfg.age_max:
                self._remove_edge(i, j)

    @staticmethod
    def _alpha(apex: float, mean: float) -> float:
        if 2 * mean >= apex:
            return 0.0
        if 3 * mean >= apex:
            return 0.5
        return 1.0

    def _merge_ok(self, l1: int, l2: int, overlap: float, stats) -> bool:
        for l in (l1, l2):
            if l not in stats:
                return True
            apex, mean = stats[l]
            if overlap >= self._alpha(apex, mean) * apex:
                return True
        return False

    def _connect_or_disconnect(self, a1: int, a2: int) -> None:
        n1, n2 = self.nodes[a1], self.nodes[a2]
        join = (
            n1.label < 0 or n2.label < 0 or n1.label == n2.label
            or self._merge_ok(n1.label, n2.label, min(n1.density, n2.density), self._sub_stats)
        )
        if join:
            self.edges[self._key(a1, a2)] = 0
            self.adj[a1].add(a2)
            self.adj[a2].add(a1)
        elif a2 in self.adj[a1]:
            self._remove_edge(a1, a2)

    def _add_density_point(self, a1: int) -> None:
        nbrs = self.adj[a1]
        if nbrs:
            dbar = float(np.mean([np.linalg.norm(self.nodes[a1].w - self.nodes[j].w) for j in nbrs]))
            self.nodes[a1].s += 1.0 / (1.0 + dbar) ** 2

    def _adapt(self, a1: int, x: np.ndarray) -> None:
        n = self.nodes[a1]
        n.wins += 1
        n.w += (x - n.w) / n.wins
        for j in self.adj[a1]:
            self.nodes[j].w += (x - self.nodes[j].w) / (100.0 * n.wins)

    def _refresh_density(self) -> None:
        for n in self.nodes.values():
            n.density = n.s / max(1, self.n_signals - n.birth)

    def _periodic_update(self) -> None:
        self._refresh_density()
        self._partition()
        if self._remove_noise():
            self._refresh_density()
            self._partition()

    def _partition(self) -> None:
        if not self.nodes:
            self._sub_stats = {}
            return
        order = sorted(self.nodes, key=lambda i: -self.nodes[i].density)
        label: dict[int, int] = {}
        nxt = 0
        for i in order:
            labelled = [j for j in self.adj[i] if j in label]
            if not labelled:
                label[i] = nxt  # local density maximum: apex of a new subclass
                nxt += 1
            else:
                label[i] = label[max(labelled, key=lambda j: self.nodes[j].density)]

        pre: dict[int, list[float]] = {}
        for i, l in label.items():
            pre.setdefault(l, []).append(self.nodes[i].density)
        pre_stats = {l: (max(v), float(np.mean(v))) for l, v in pre.items()}

        overlap: dict[tuple[int, int], float] = {}
        for (i, j) in self.edges:
            li, lj = label[i], label[j]
            if li != lj:
                pair = (min(li, lj), max(li, lj))
                overlap[pair] = max(overlap.get(pair, -1.0), min(self.nodes[i].density, self.nodes[j].density))

        parent = {l: l for l in pre}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for (la, lb), ov in overlap.items():
            if self._merge_ok(la, lb, ov, pre_stats):
                parent[find(la)] = find(lb)

        for (i, j) in list(self.edges):
            if find(label[i]) != find(label[j]):
                self._remove_edge(i, j)

        renumber: dict[int, int] = {}
        for i in order:
            root = find(label[i])
            renumber.setdefault(root, len(renumber))
            self.nodes[i].label = renumber[root]

        final: dict[int, list[float]] = {}
        for n in self.nodes.values():
            final.setdefault(n.label, []).append(n.density)
        self._sub_stats = {l: (max(v), float(np.mean(v))) for l, v in final.items()}

    def _remove_noise(self) -> bool:
        if len(self.nodes) <= 2:
            return False
        mean_density = float(np.mean([n.density for n in self.nodes.values()]))
        doomed = []
        for i, n in self.nodes.items():
            deg = len(self.adj[i])
            if deg == 0 or (deg == 1 and n.density < self.cfg.c1 * mean_density) \
                    or (deg == 2 and n.density < self.cfg.c2 * mean_density):
                doomed.append(i)
        removed = False
        for i in doomed:
            if len(self.nodes) > 2:
                self._remove_node(i)
                removed = True
        return removed
