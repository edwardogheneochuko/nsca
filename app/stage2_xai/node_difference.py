"""Stage 2 – Node difference ("patient vs winner node").

ESOINN's own explanation.  Each symptom of the patient is compared with the
winner node's prototype and with the nearest node from a different cluster
(the runner-up), giving a symptom-by-symptom differential.
"""
from __future__ import annotations

import numpy as np

TYPICAL = 0.5   # prototype weight at which a symptom counts as typical of the node
RARE = 0.1      # prototype weight below which a symptom counts as unusual for the node


def explain(x: np.ndarray, w_win: np.ndarray, w_run: np.ndarray | None, meanings: list[str], top: int = 8) -> dict:
    x = x.astype(np.float32)

    def item(f):
        d = {"feature_index": int(f), "meaning": meanings[f], "patient": int(x[f]),
             "winner_weight": round(float(w_win[f]), 3)}
        if w_run is not None:
            d["runner_up_weight"] = round(float(w_run[f]), 3)
        return d

    present = np.flatnonzero(x > 0)
    supporting = [item(f) for f in present[np.argsort(-w_win[present])] if w_win[f] >= TYPICAL]
    unexpected = [item(f) for f in present if w_win[f] < RARE]
    expected_missing = np.flatnonzero((x == 0) & (w_win >= TYPICAL))
    missing = [item(f) for f in expected_missing[np.argsort(-w_win[expected_missing])]]

    differential = []
    if w_run is not None:
        # positive score = this symptom makes the patient look more like the winner than the runner-up
        score = np.abs(x - w_run) - np.abs(x - w_win)
        cand = np.flatnonzero(np.abs(score) > 0.2)
        for f in cand[np.argsort(-np.abs(score[cand]))][:top]:
            d = item(f)
            d["favours"] = "winner" if score[f] > 0 else "runner_up"
            d["score"] = round(float(score[f]), 3)
            differential.append(d)

    n_sup = len(supporting)
    return {"supporting": supporting[:top], "unexpected": unexpected[:top], "expected_but_missing": missing[:top],
            "differential_vs_runner_up": differential,
            "support_ratio": round(n_sup / max(len(present), 1), 3)}
