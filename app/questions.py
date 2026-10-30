"""Questionnaire for the frontend and DDXPlus default answers.

DDXPlus symptoms are questions.  Some have answer options (pain location, pain
intensity 0-10, travel region), and follow-up questions are only asked when their
parent is answered "yes".  In the training data a follow-up is ALWAYS answered once
its parent is "yes" (with a default value when the patient picked nothing), and
travel is always recorded (default "No").  `complete_defaults` reproduces that, so
the app only needs to send what the patient actually chose.

For the other datasets every symptom column is a simple yes/no question.
"""
from __future__ import annotations

import json

import numpy as np

from app.config import DATA_DIR

EVIDENCES = DATA_DIR / "ddxplus" / "release_evidences.json"


def build(features: list[str], meanings: list[str], is_antecedent, dataset: str) -> list[dict]:
    fi = {f: i for i, f in enumerate(features)}
    if dataset != "ddxplus" or not EVIDENCES.exists():
        return [{"id": f, "text": m, "type": "yes_no", "kind": "risk_factor" if a else "symptom",
                 "parent": None, "options": [{"value": f, "label": "Yes"}], "default": None}
                for f, m, a in zip(features, meanings, is_antecedent)]
    ev = json.loads(EVIDENCES.read_text())
    qs = []
    for k, v in ev.items():
        parent = v.get("code_question", k)
        parent = None if parent == k else parent
        kind = "risk_factor" if v.get("is_antecedent") else "symptom"
        if v["data_type"] == "B":
            if k not in fi:
                continue
            qtype, options, default = "yes_no", [{"value": k, "label": "Yes"}], None
        else:
            options = []
            for val in v["possible-values"]:
                col = f"{k}={val}"
                if col in fi:
                    label = meanings[fi[col]].split(" = ", 1)[-1]
                    options.append({"value": col, "label": label})
            if not options:
                continue
            numeric = all(isinstance(val, (int, float)) for val in v["possible-values"])
            qtype = "scale" if numeric else ("multi" if v["data_type"] == "M" else "single")
            dcol = f"{k}={v['default_value']}"
            default = dcol if dcol in fi else None
        qs.append({"id": k, "text": v["question_en"], "type": qtype, "kind": kind, "parent": parent,
                   "options": options, "default": default})
    # parents first, each followed by its follow-ups
    by_parent: dict[str | None, list[dict]] = {}
    for q in qs:
        by_parent.setdefault(q["parent"], []).append(q)
    ordered = []
    for q in by_parent.get(None, []):
        ordered.append(q)
        ordered.extend(by_parent.get(q["id"], []))
    return ordered


def complete_defaults(x: np.ndarray, questions: list[dict], features: list[str]) -> list[str]:
    """Fill DDXPlus default answers in place; returns the columns that were filled."""
    fi = {f: i for i, f in enumerate(features)}
    filled = []
    for q in questions:
        if q["type"] == "yes_no" or not q["default"]:
            continue
        answered = any(x[fi[o["value"]]] for o in q["options"])
        if answered:
            continue
        if q["parent"] is None or (q["parent"] in fi and x[fi[q["parent"]]]):
            x[fi[q["default"]]] = 1
            filled.append(q["default"])
    return filled
