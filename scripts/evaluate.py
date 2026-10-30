"""Evaluate the trained pipeline on the held-back test patients.

    python -m scripts.evaluate --dataset ddxplus

Test patients include diseases ESOINN and the knowledge base never saw (held out).
Reports Stage 1 cluster quality, Stage 2 fidelity/reliability, and Stage 3 outcomes
(known-disease accuracy, false "new" rate, held-out detection rate).
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import Counter

import numpy as np

from app.config import REPORT_DIR
from app.data.loaders import load
from app.engine import Engine
from app.stage1_neural.metrics import stage1_report


def evaluate(name: str, log=print) -> dict:
    d = load(name)
    e = Engine(name)
    known = ~d.test_is_new
    res = []
    for i, x in enumerate(d.X_test):
        r = e.identify(x, record_flag=False)
        rec = r["stage2_fused_record"]
        cands = r["stage3_symbolic"]["candidates"]
        res.append({"outcome": r["outcome"], "disease": r["disease"], "confidence": r["confidence"],
                    "reliable": rec["reliable"], "agreement": rec["agreement"], "outlier": rec["stage1_outlier"],
                    "cluster": rec["cluster_id"], "cands": [c["disease"] for c in cands if c["source"] == "ontology top-k"],
                    "consistent": [c["disease"] for c in cands if c["consistent"]]})
        if log and (i + 1) % 3000 == 0:
            log(f"[{name}] {i + 1}/{len(d.X_test)}")
    y = d.y_test
    out = {"dataset": name, "test_patients": int(len(y)), "known_test_patients": int(known.sum()),
           "heldout_test_patients": int((~known).sum()), "heldout_diseases": d.heldout_diseases}

    s1 = stage1_report(e.esoinn, d.X_test[known], y[known], len(d.known_diseases))
    outl = np.array([r["outlier"] for r in res])
    s1["outlier_rate_known"] = round(float(outl[known].mean()), 4)
    s1["outlier_rate_heldout"] = round(float(outl[~known].mean()), 4)
    out["stage1"] = s1

    agree = np.array([r["agreement"] for r in res]); rel = np.array([r["reliable"] for r in res])
    out["stage2"] = {"fidelity_validation": e.surrogate.fidelity,
                     "cart_esoinn_agreement_test": round(float(agree.mean()), 4),
                     "reliable_rate_known": round(float(rel[known].mean()), 4),
                     "reliable_rate_heldout": round(float(rel[~known].mean()), 4)}

    outcome = np.array([r["outcome"] for r in res]); dis = np.array([str(r["disease"]) for r in res])
    flag = outcome == "flag_as_new"
    correct = (outcome == "known_disease") & (dis == y)
    in_topk = np.array([yy in r["cands"] for yy, r in zip(y, res)])
    conf = Counter((r["confidence"], bool(c)) for r, c, k in zip(res, correct, known) if k and r["outcome"] == "known_disease")
    s3 = {"known_accuracy": round(float(correct[known].mean()), 4),
          "known_accuracy_when_not_flagged": round(float(correct[known & ~flag].sum() / max((known & ~flag).sum(), 1)), 4),
          "true_disease_in_top3": round(float(in_topk[known].mean()), 4),
          "false_new_rate_known": round(float(flag[known].mean()), 4),
          "heldout_flagged_as_new": round(float(flag[~known].mean()), 4),
          "accuracy_by_confidence": {f"{c}": {"n": conf[(c, True)] + conf[(c, False)],
                                              "accuracy": round(conf[(c, True)] / max(conf[(c, True)] + conf[(c, False)], 1), 4)}
                                     for c in ("high", "medium", "low")}}
    # Variant: also flag when Stage 1 says outlier AND the record is unreliable
    flag_b = flag | (outl & ~rel)
    s3["variant_stage1_outlier_and_unreliable_also_flags"] = {
        "false_new_rate_known": round(float(flag_b[known].mean()), 4),
        "heldout_flagged_as_new": round(float(flag_b[~known].mean()), 4)}
    per = {}
    for h in d.heldout_diseases:
        m = y == h
        cl = Counter(int(r["cluster"]) for r, f in zip(np.array(res)[m], flag[m]) if f)
        wrong = Counter(r["disease"] for r, f in zip(np.array(res)[m], flag[m]) if not f)
        per[h] = {"patients": int(m.sum()), "flagged": round(float(flag[m].mean()), 4),
                  "flagged_cases_spread_over_clusters": len(cl),
                  "most_common_wrong_known_diagnosis": wrong.most_common(2)}
    s3["per_heldout_disease"] = per
    if d.test_differential is not None:
        overlap, path_in = [], []
        for r, diff, k, yy in zip(res, d.test_differential, known, y):
            if not k:
                continue
            dd = {p for p, _ in ast.literal_eval(diff)}
            overlap.append(np.mean([c in dd for c in r["cands"]]) if r["cands"] else 0)
        s3["ddxplus_top3_candidates_in_patient_differential"] = round(float(np.mean(overlap)), 4)
    out["stage3"] = s3
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"evaluation_{name}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "ddxplus", "phenopackets", "all"], default="all")
    a = ap.parse_args()
    for n in (["mendeley", "ddxplus", "phenopackets"] if a.dataset == "all" else [a.dataset]):
        print(json.dumps(evaluate(n), indent=1, default=str))
