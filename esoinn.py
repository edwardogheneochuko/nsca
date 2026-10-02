"""Stage 1: ESOINN (Furao, Ogura & Hasegawa, 2007) and a stacked two-layer wrapper.

Simplifications (state these in the thesis): density uses periods since a node's
birth; the merge test uses the alpha rule from the paper on the two subclasses.
"""
import numpy as np
from collections import defaultdict

class ESOINN:
    def __init__(self, omega, lam=200, age_max=50, c1=0.001, c2=1.0, seed=0):
        self.omega = np.asarray(omega, float)
        self.lam, self.age_max, self.c1, self.c2 = lam, age_max, c1, c2
        self.w, self.M, self.acc, self.birth, self.label = {}, {}, {}, {}, {}
        self.adj = defaultdict(set)
        self.age = {}
        self.t, self._next = 0, 0
        self.events = []                      # (t, "new_node" | "delete_node" | ...)

    # ---------- helpers ----------
    def _dist(self, x, m, W):
        wt = self.omega * m
        return np.sqrt((((W - x) ** 2) * wt).sum(1) / max(wt.sum(), 1e-9))

    def _stack(self):
        ids = list(self.w)
        return ids, np.stack([self.w[i] for i in ids])

    def _ek(self, i, j): return (i, j) if i < j else (j, i)

    def _add(self, x):
        i = self._next; self._next += 1
        self.w[i] = x.copy(); self.M[i] = 1; self.acc[i] = 0.0
        self.birth[i] = self.t; self.label[i] = -1
        self.events.append((self.t, "new_node", i)); return i

    def _link(self, i, j):
        self.adj[i].add(j); self.adj[j].add(i); self.age[self._ek(i, j)] = 0

    def _unlink(self, i, j):
        self.adj[i].discard(j); self.adj[j].discard(i); self.age.pop(self._ek(i, j), None)

    def _threshold(self, i, ids, W):
        ones = np.ones_like(self.omega)
        if self.adj[i]:
            Wn = np.stack([self.w[n] for n in self.adj[i]])
            return self._dist(self.w[i], ones, Wn).max()
        d = self._dist(self.w[i], ones, W); d[ids.index(i)] = np.inf
        return d.min()

    def density(self, i):
        periods = max(1.0, (self.t - self.birth[i]) / self.lam)
        return self.acc[i] / periods

    def _alpha_max(self, lab):
        hs = np.array([self.density(i) for i in self.w if self.label[i] == lab])
        if len(hs) == 0: return 0.0, 0.0
        mean, mx = hs.mean(), hs.max()
        a = 0.0 if 2 * mean >= mx else (0.5 if 3 * mean >= mx else 1.0)
        return a, mx

    def _can_merge(self, i, j):
        hmin = min(self.density(i), self.density(j))
        for lab in (self.label[i], self.label[j]):
            a, mx = self._alpha_max(lab)
            if hmin > a * mx: return True
        return False

    # ---------- online learning ----------
    def learn(self, x, m=None):
        x = np.asarray(x, float); m = np.ones_like(x) if m is None else np.asarray(m, float)
        self.t += 1
        if len(self.w) < 2:
            self._add(x); return "new_node"
        ids, W = self._stack()
        d = self._dist(x, m, W); o = np.argsort(d)
        s1, s2 = ids[o[0]], ids[o[1]]
        if d[o[0]] > self._threshold(s1, ids, W) or d[o[1]] > self._threshold(s2, ids, W):
            self._add(x); event = "new_node"
        else:
            event = "adapt"
            for n in list(self.adj[s1]): self.age[self._ek(s1, n)] += 1
            l1, l2 = self.label[s1], self.label[s2]
            if l1 == -1 or l2 == -1 or l1 == l2 or self._can_merge(s1, s2):
                self._link(s1, s2)
            else:
                self._unlink(s1, s2)
            if self.adj[s1]:
                Wn = np.stack([self.w[n] for n in self.adj[s1]])
                dbar = self._dist(self.w[s1], np.ones_like(x), Wn).mean()
            else:
                dbar = d[o[0]]
            self.acc[s1] += 1.0 / (1.0 + dbar) ** 2
            self.M[s1] += 1
            self.w[s1] += (x - self.w[s1]) / self.M[s1]
            for n in self.adj[s1]:
                self.w[n] += (x - self.w[n]) / (100.0 * self.M[s1])
            for (a, b), ag in list(self.age.items()):
                if ag > self.age_max: self._unlink(a, b)
        if self.t % self.lam == 0: self._periodic()
        return event

    def _periodic(self):
        ids = list(self.w)
        h = {i: self.density(i) for i in ids}
        # 1. subclasses: follow uphill pointers to apexes
        def apex(i):
            while True:
                up = [n for n in self.adj[i] if h[n] > h[i]]
                if not up: return i
                i = max(up, key=lambda n: h[n])
        for i in ids: self.label[i] = apex(i)
        # 2. cross-subclass edges: merge or cut
        for (a, b) in list(self.age):
            if self.label[a] != self.label[b]:
                if self._can_merge(a, b):
                    old, new = self.label[b], self.label[a]
                    for i in ids:
                        if self.label[i] == old: self.label[i] = new
                else:
                    self._unlink(a, b)
        # 3. noise removal
        hbar = np.mean(list(h.values()))
        for i in ids:
            k = len(self.adj[i])
            if k == 0 or (k == 1 and h[i] < self.c2 * hbar) or (k == 2 and h[i] < self.c1 * hbar):
                if len(self.w) > 2: self._delete(i)

    def _delete(self, i):
        for n in list(self.adj[i]): self._unlink(i, n)
        for d in (self.w, self.M, self.acc, self.birth, self.label): d.pop(i, None)
        self.adj.pop(i, None)
        self.events.append((self.t, "delete_node", i))

    # ---------- read-out ----------
    def clusters(self):
        """Connected components of the node graph -> {node_id: cluster_id}."""
        seen, out, c = set(), {}, 0
        for s in self.w:
            if s in seen: continue
            stack = [s]
            while stack:
                u = stack.pop()
                if u in seen: continue
                seen.add(u); out[u] = c; stack.extend(self.adj[u] - seen)
            c += 1
        return out

    def winner(self, x, m=None):
        x = np.asarray(x, float); m = np.ones_like(x) if m is None else np.asarray(m, float)
        ids, W = self._stack(); d = self._dist(x, m, W); k = int(np.argmin(d))
        return ids[k], float(d[k])

    def membership(self, x, m=None, tau=0.1):
        """Soft scores mu_k(x) = softmax(-d(x, C_k)/tau), d = distance to nearest node of C_k."""
        x = np.asarray(x, float); m = np.ones_like(x) if m is None else np.asarray(m, float)
        ids, W = self._stack(); d = self._dist(x, m, W); cl = self.clusters()
        ks = sorted(set(cl.values()))
        dk = np.array([min(d[j] for j, i in enumerate(ids) if cl[i] == k) for k in ks])
        z = np.exp(-(dk - dk.min()) / tau); return dict(zip(ks, z / z.sum()))


class TwoLayerESOINN:
    """L1: presentation prototypes. L2: disease clusters learned from L1 prototypes."""
    def __init__(self, omega, l1=dict(lam=100, age_max=25), l2=dict(lam=100, age_max=50), seed=0):
        self.l1, self.l2 = ESOINN(omega, **l1), ESOINN(omega, **l2)
        self.rng = np.random.default_rng(seed)

    def fit(self, X, M=None, epochs=1):
        X = np.asarray(X, float); M = np.ones_like(X) if M is None else np.asarray(M, float)
        for _ in range(epochs):
            for k in self.rng.permutation(len(X)): self.l1.learn(X[k], M[k])
        self.refresh_l2()
        return self

    def refresh_l2(self, max_rep=5):
        ids = list(self.l1.w)
        h = np.array([self.l1.density(i) for i in ids]); hmax = max(h.max(), 1e-9)
        replay = [self.l1.w[i] for i, hi in zip(ids, h) for _ in range(1 + int(round(max_rep * hi / hmax)))]
        self.l2 = ESOINN(self.l1.omega, self.l2.lam, self.l2.age_max)
        for k in self.rng.permutation(len(replay)): self.l2.learn(replay[k])

    def assign(self, x, m=None):
        n1, d1 = self.l1.winner(x, m); n2, d2 = self.l2.winner(x, m)
        mu = self.l2.membership(x, m); cl = self.l2.clusters()
        return dict(l1_node=n1, l2_node=n2, l2_cluster=cl[n2], membership=mu)
