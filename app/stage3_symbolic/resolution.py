"""Stage 3 – Resolution ("known or new") -> Known disease | Flag as new.

Rules, reading the fused record's reliability mark:
  * no consistent candidate                       -> flag_as_new (provisional)
  * record reliable and the ESOINN cluster's
    disease is consistent                         -> known_disease, high confidence
  * otherwise, best-scoring consistent candidate  -> known_disease; confidence is
    medium when the record is reliable, low when it is not (the neural suggestion
    is then ignored and the decision rests on the symbolic checks alone)
"""
from __future__ import annotations


def resolve(checked: list[dict], record) -> dict:
    consistent = [c for c in checked if c["consistent"]]
    if not consistent:
        return {"outcome": "flag_as_new", "disease": None, "confidence": "provisional",
                "reason": "no known disease passes the consistency checks",
                "stage1_outlier": record.stage1_outlier, "record_reliable": record.reliable}
    best = max(consistent, key=lambda c: c["score"])
    if record.reliable:
        hint = next((c for c in consistent if c["disease"] == record.cluster_name), None)
        if hint is not None:
            return {"outcome": "known_disease", "disease": hint["disease"], "confidence": "high",
                    "reason": "ESOINN cluster, surrogate rule and knowledge-base checks agree",
                    "stage1_outlier": record.stage1_outlier, "record_reliable": True}
        return {"outcome": "known_disease", "disease": best["disease"], "confidence": "medium",
                "reason": "record reliable but the cluster's disease failed the checks; best consistent candidate used",
                "stage1_outlier": record.stage1_outlier, "record_reliable": True}
    return {"outcome": "known_disease", "disease": best["disease"], "confidence": "low",
            "reason": "explanation record unreliable (" + "; ".join(record.unreliable_reasons) +
                      "); decided on symbolic checks only",
            "stage1_outlier": record.stage1_outlier, "record_reliable": False}
