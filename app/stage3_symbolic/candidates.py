"""Stage 3 – Candidates ("top-k diseases").

Scores every known disease in the knowledge base against the patient's grounded
HPO terms and returns the top k.  Terms that are parent or child of a disease term
in the HPO hierarchy earn half credit, so a patient with "Chest pain" still
partly matches a disease profiled with "Pleuritic chest pain".
"""
from __future__ import annotations

import numpy as np

PARTIAL_CREDIT = 0.5


def related_matrix(kb, hpo) -> np.ndarray:
    """R[i, j] = 1 if term i is a direct parent or child of term j in HPO (identity excluded)."""
    n = len(kb.terms)
    R = np.zeros((n, n), np.float32)
    for i, t in enumerate(kb.terms):
        if not t.startswith("HP:"):
            continue
        rel = set(hpo.terms.get(t, {}).get("is_a", [])) | set(hpo.children.get(t, []))
        for r in rel:
            j = kb.index.get(r)
            if j is not None:
                R[i, j] = 1
    return R


def score_all(t_patient: np.ndarray, kb, R: np.ndarray) -> dict:
    """Vectorised scores of one patient (term vector) against every disease."""
    p = t_patient.astype(np.float32)
    n = max(p.sum(), 1.0)
    # effective frequency of each patient term for each disease, with partial credit via the hierarchy
    partial = (kb.F @ R.T) > 0                              # disease has a related term
    Fp = np.maximum(kb.F, PARTIAL_CREDIT * partial * (kb.F @ R.T).clip(0, 1))
    match = (Fp * p).sum(1) / n                             # how typical the patient's terms are for D
    core_n = kb.core.sum(1)
    covered = (kb.core & (p > 0)).sum(1)
    coverage = np.where(core_n > 0, covered / np.maximum(core_n, 1), 1.0)
    score = np.where(match + coverage > 0, 2 * match * coverage / np.maximum(match + coverage, 1e-9), 0.0)
    return {"score": score, "match": match, "coverage": coverage, "Fp": Fp}


def top_k(t_patient, kb, R, k: int = 3, hint: str | None = None) -> list[dict]:
    s = score_all(t_patient, kb, R)
    order = list(np.argsort(-s["score"])[:k])
    hint_i = kb.diseases.index(hint) if hint in kb.diseases else None
    if hint_i is not None and hint_i not in order:
        order.append(hint_i)                                # neural suggestion is always examined
    out = []
    for i in order:
        out.append({"disease": kb.diseases[i], "score": round(float(s["score"][i]), 4),
                    "match": round(float(s["match"][i]), 4), "core_coverage": round(float(s["coverage"][i]), 4),
                    "source": "ontology top-k" if i in order[:k] else "ESOINN cluster suggestion",
                    "_i": int(i)})
    return out
