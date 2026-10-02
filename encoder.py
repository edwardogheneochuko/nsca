"""Stage 0: symptom encoding. Raw symptoms -> (vector, mask) with IDF weights."""
import numpy as np

SEVERITY = {"absent": 0.0, "mild": 0.33, "moderate": 0.66, "severe": 1.0}

class SymptomEncoder:
    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.index = {s: i for i, s in enumerate(self.vocab)}
        self.omega = np.ones(len(self.vocab))          # IDF weights

    def fit_idf(self, X):
        """Rare symptoms get a larger weight (e.g. haemorrhagic rash > headache)."""
        X = np.asarray(X, float)
        df = (X > 0).sum(0)
        self.omega = np.log((1 + len(X)) / (1 + df)) + 1.0
        return self

    def encode(self, present, absent=None, asked_all=True):
        """present: list of symptoms or {symptom: 'mild'|'moderate'|'severe'|float}.
        absent: symptoms explicitly denied. If asked_all is False, anything not
        mentioned is 'not asked' (mask=0) instead of 'absent'."""
        x = np.zeros(len(self.vocab))
        m = np.ones(len(self.vocab)) if asked_all else np.zeros(len(self.vocab))
        items = present.items() if isinstance(present, dict) else [(s, 1.0) for s in present]
        for s, v in items:
            if s in self.index:
                x[self.index[s]] = SEVERITY.get(v, v) if isinstance(v, str) else float(v)
                m[self.index[s]] = 1.0
        for s in (absent or []):
            if s in self.index:
                m[self.index[s]] = 1.0
        return x, m
