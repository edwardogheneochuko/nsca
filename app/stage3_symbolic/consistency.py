"""Stage 3 – Consistency ("exclusion rules").

Each candidate is tested against the evidence:
  * required symptoms present   – share of the disease's core terms the patient has
  * excluding symptoms absent   – terms never seen with the disease but typical of another
  * unexplained symptoms        – patient terms never seen with the disease
  * node-difference alignment   – share of Stage 2's supporting symptoms typical of the disease

Thresholds are calibrated on validation patients of KNOWN diseases only, so the
held-out diseases play no part in setting them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from app.stage3_symbolic.knowledge_base import TYPICAL


@dataclass
class Thresholds:
    min_score: float = 0.3
    min_core_coverage: float = 0.3
    max_excluding: int = 0
    max_unexplained: float = 0.3

    def to_dict(self):
        return asdict(self)


def check(t_patient: np.ndarray, cand: dict, kb, Fp: np.ndarray, th: Thresholds,
          supporting_terms: list[int] | None = None) -> dict:
    i = cand["_i"]
    present = np.flatnonzero(t_patient)
    excl_hits = [kb.terms[j] for j in present if kb.excl[i, j]]
    unexplained = [kb.terms[j] for j in present if Fp[i, j] == 0]
    unexp_frac = len(unexplained) / max(len(present), 1)
    missing_core = [kb.terms[j] for j in np.flatnonzero(kb.core[i] & (t_patient == 0))]
    align = None
    if supporting_terms:
        align = round(float(np.mean([Fp[i, j] >= TYPICAL for j in supporting_terms])), 3)
    failures = []
    if cand["score"] < th.min_score:
        failures.append(f"match score {cand['score']:.2f} < {th.min_score:.2f}")
    if cand["core_coverage"] < th.min_core_coverage:
        failures.append(f"core symptoms covered {cand['core_coverage']:.2f} < {th.min_core_coverage:.2f}")
    if len(excl_hits) > th.max_excluding:
        failures.append(f"{len(excl_hits)} excluding symptom(s) present")
    if unexp_frac > th.max_unexplained:
        failures.append(f"{unexp_frac:.0%} of symptoms never seen with this disease")
    # Structured form of the four checks, for plain-language display
    checks = [
        {"name": "overall_match", "passed": bool(cand["score"] >= th.min_score),
         "value": round(float(cand["score"]), 3), "limit": th.min_score},
        {"name": "usual_symptoms", "passed": bool(cand["core_coverage"] >= th.min_core_coverage),
         "value": round(float(cand["core_coverage"]), 3), "limit": th.min_core_coverage},
        {"name": "nothing_against", "passed": bool(len(excl_hits) <= th.max_excluding),
         "value": len(excl_hits), "limit": th.max_excluding},
        {"name": "symptoms_explained", "passed": bool(unexp_frac <= th.max_unexplained),
         "value": round(float(unexp_frac), 3), "limit": th.max_unexplained},
    ]
    core_present = [kb.label(kb.terms[j]) for j in present if kb.core[i, j]]
    typical_present = [kb.label(kb.terms[j]) for j in present if not kb.core[i, j] and Fp[i, j] >= 0.3]
    return {**{k: v for k, v in cand.items() if k != "_i"},
            "consistent": not failures, "failed_checks": failures, "checks": checks,
            "core_present": core_present, "typical_present": typical_present,
            "excluding_present": [kb.label(t) for t in excl_hits],
            "unexplained_symptoms": [kb.label(t) for t in unexplained],
            "missing_core_symptoms": [kb.label(t) for t in missing_core][:8],
            "node_difference_alignment": align}


def calibrate(T_val: np.ndarray, y_val: np.ndarray, kb, R, quantile: float = 0.05) -> Thresholds:
    """Set each threshold so that about 95 % of known validation patients pass it for their own disease."""
    from app.stage3_symbolic.candidates import score_all
    scores, covs, unexp, excl = [], [], [], []
    for t, d in zip(T_val, y_val):
        if d not in kb.diseases:
            continue
        i = kb.diseases.index(d)
        s = score_all(t, kb, R)
        present = np.flatnonzero(t)
        scores.append(s["score"][i]); covs.append(s["coverage"][i])
        unexp.append(np.mean(s["Fp"][i, present] == 0) if len(present) else 0.0)
        excl.append(int(kb.excl[i, present].sum()))
    return Thresholds(min_score=round(float(np.quantile(scores, quantile)), 4),
                      min_core_coverage=round(float(np.quantile(covs, quantile)), 4),
                      max_excluding=int(np.quantile(excl, 1 - quantile)),
                      max_unexplained=round(float(np.quantile(unexp, 1 - quantile)), 4))
