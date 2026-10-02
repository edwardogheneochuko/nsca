"""Stage 2: XAI. KernelSHAP (local+global), IMM threshold tree (global rules), greedy counterfactuals."""
import numpy as np, shap

class Explainer:
    def __init__(self, net, X, vocab, tau=0.03):
        self.X, self.vocab, self.tau, self.omega = np.asarray(X, float), list(vocab), tau, net.l2.omega
        l2 = net.l2; ids = list(l2.w); self.W = np.stack([l2.w[i] for i in ids])
        cl = l2.clusters(); self.cl_of = np.array([cl[i] for i in ids]); self.K = len(set(cl.values()))
        self.assign = self.mu(self.X).argmax(1)
        self.centers = np.stack([self.X[self.assign == k].mean(0) if (self.assign == k).any()
                                 else self.W[self.cl_of == k].mean(0) for k in range(self.K)])
        self.tree = self._imm(); self.sig = {}

    def mu(self, Z):
        """Soft membership mu_k(x) = softmax(-d(x, C_k)/tau)."""
        Z = np.atleast_2d(Z)
        D = np.sqrt((((Z[:, None, :] - self.W[None]) ** 2) * self.omega).sum(2) / self.omega.sum())
        dk = np.stack([D[:, self.cl_of == k].min(1) for k in range(self.K)], 1)
        e = np.exp(-(dk - dk.min(1, keepdims=True)) / self.tau); return e / e.sum(1, keepdims=True)

    # ---- XAI-1: KernelSHAP ----
    def shap_local(self, x, k, n=8, nsamples=300):
        ex = shap.KernelExplainer(lambda Z: self.mu(Z)[:, k], self.W)
        phi = np.asarray(ex.shap_values(x[None], nsamples=nsamples, silent=True)).reshape(-1)
        return [(self.vocab[j], float(phi[j])) for j in np.argsort(-np.abs(phi))[:n] if abs(phi[j]) > 1e-4]

    def signature(self, k, n=8, per=6):
        """Global: mean |SHAP| over sample patients of cluster k."""
        if k not in self.sig:
            P = self.X[self.assign == k][:per]
            if len(P) == 0: self.sig[k] = []; return []
            ex = shap.KernelExplainer(lambda Z: self.mu(Z)[:, k], self.W)
            phi = np.abs(np.asarray(ex.shap_values(P, nsamples=100, silent=True))).reshape(len(P), -1).mean(0)
            self.sig[k] = [(self.vocab[j], float(phi[j])) for j in np.argsort(-phi)[:n] if phi[j] > 1e-4]
        return self.sig[k]

    # ---- XAI-2: counterfactuals (greedy sparse search; lightweight stand-in for DiCE) ----
    def counterfactual(self, x, k_from, k_to, max_changes=3):
        cur, changes = x.copy(), []
        for _ in range(max_changes):
            cand = np.repeat(cur[None], len(x), 0); i = np.arange(len(x)); cand[i, i] = 1 - cand[i, i]
            P = self.mu(cand); j = int(np.argmax(P[:, k_to] - P[:, k_from])); cur = cand[j]
            changes.append((self.vocab[j], "add" if cur[j] == 1 else "remove"))
            if self.mu(cur)[0].argmax() == k_to: return dict(to=k_to, valid=True, changes=changes)
        return dict(to=k_to, valid=False, changes=changes)

    # ---- XAI-3: IMM threshold tree (Moshkovitz et al., 2020), adapted to binary symptoms ----
    def _imm(self):
        X, A, C = self.X, self.assign, self.centers
        def build(P, Cs):
            if len(Cs) == 1: return dict(leaf=int(Cs[0]))
            best = None
            for j in range(X.shape[1]):
                vals = np.unique(C[Cs, j])
                for th in (vals[:-1] + vals[1:]) / 2:
                    R = [c for c in Cs if C[c, j] > th]; L = [c for c in Cs if C[c, j] <= th]
                    if not R or not L: continue
                    m = int(((X[P, j] > .5) != np.isin(A[P], R)).sum())
                    if best is None or m < best[0]: best = (m, j, L, R)
            if best is None: return dict(leaf=int(Cs[0]))
            _, j, L, R = best; px = X[P, j] > .5
            return dict(f=j, name=self.vocab[j],
                        no=build(P[~px & np.isin(A[P], L)], L), yes=build(P[px & np.isin(A[P], R)], R))
        t = build(np.arange(len(X)), list(range(self.K)))
        self.fidelity = float(np.mean([self.path(x, t)[1] == a for x, a in zip(X, A)]))
        return t

    def path(self, x, t=None):
        t = t or self.tree; p = []
        while "leaf" not in t:
            pres = bool(x[t["f"]] > .5); p.append((t["f"], pres)); t = t["yes" if pres else "no"]
        return p, t["leaf"]
