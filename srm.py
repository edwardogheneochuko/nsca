"""Stage 3: Symbolic reasoning. Disease-symptom KB, grounding score, red-flag and unmapped-cluster rules."""
import numpy as np, json, os

RED_FLAGS = {"chest_pain", "coma", "altered_sensorium", "blood_in_sputum", "weakness_of_one_body_side",
             "slurred_speech", "loss_of_balance", "stomach_bleeding", "bloody_stool", "visual_disturbances"}

class SRM:
    def __init__(self, X, y, vocab, omega, desc, prec, codes_file="data/codes.json", beta=0.6, tau_g=0.4):
        self.vocab, self.omega, self.beta, self.tau_g = list(vocab), omega, beta, tau_g
        y = np.asarray(y); self.dis = sorted(set(y))
        self.P = np.stack([X[y == d].mean(0) for d in self.dis])          # KB: symptom prevalence per disease
        self.desc, self.prec = desc.to_dict(), prec
        self.codes = json.load(open(codes_file)) if os.path.exists(codes_file) else {}
        self.red = [s for s in self.vocab if s in RED_FLAGS]
        self.cands = {}

    def wcos(self, x):
        num = (self.P * x * self.omega).sum(1)
        return num / (np.sqrt((x * x * self.omega).sum()) * np.sqrt((self.P ** 2 * self.omega).sum(1)) + 1e-9)

    def ground_cluster(self, k, centroid, top=8):
        """Step 1: cluster grounding - match the cluster centroid to KB diseases."""
        s = self.wcos(centroid); self.cands[k] = list(np.argsort(-s)[:top])
        return [(self.dis[i], float(s[i])) for i in self.cands[k][:3]]

    def diagnose(self, x, k, path):
        """Steps 2-4: score candidates of the cluster with the patient's findings and rule path."""
        cos, out = self.wcos(x), []
        for i in self.cands[k]:
            cov = np.mean([(self.P[i, j] >= .3) == pres for j, pres in path]) if path else 1.0
            S = self.beta * cos[i] + (1 - self.beta) * cov
            d = self.dis[i]
            out.append(dict(disease=d, score=float(S), cosine=float(cos[i]), rule_coverage=float(cov),
                            matched=[self.vocab[j] for j in np.where((x > 0) & (self.P[i] >= .3))[0]],
                            description=self.desc.get(d, ""),
                            precautions=(self.prec.loc[d].dropna().tolist() if d in self.prec.index else []),
                            codes=self.codes.get(d, {"snomed": None, "icd11": None})))
        out.sort(key=lambda r: -r["score"]); tot = sum(r["score"] for r in out) or 1
        for r in out: r["share"] = r["score"] / tot
        return dict(status="unmapped" if out[0]["score"] < self.tau_g else "grounded", ranked=out[:5])

    def red_flags(self, symptoms): return [s for s in symptoms if s in self.red]
