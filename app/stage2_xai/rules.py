"""Stage 2 – Rule extraction ("tree paths to if-then").

Each root-to-leaf path of the CART surrogate becomes an if-then rule such as
"IF fever AND cough AND NOT sore throat THEN cluster 7".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def rule_text(clauses: list[dict], cluster: int, max_absent: int = 3) -> str:
    """Readable form: every present symptom, the first few absent ones, and a count of the rest."""
    pos = [c["meaning"] for c in clauses if c["present"]]
    neg = [c["meaning"] for c in clauses if not c["present"]]
    parts = pos + [f"NOT {m}" for m in neg[:max_absent]]
    if len(neg) > max_absent:
        parts.append(f"{len(neg) - max_absent} other symptoms absent")
    return f"IF {' AND '.join(parts) or 'TRUE'} THEN cluster {cluster}"


def _leaf_stats(tree):
    t = tree.tree_
    value = t.value[:, 0, :]
    purity = value.max(1) / np.maximum(value.sum(1), 1e-9)
    return t, purity


def rule_for_patient(surrogate, x: np.ndarray, meanings: list[str]) -> dict:
    tree = surrogate.tree
    t, purity = _leaf_stats(tree)
    path = tree.decision_path(x.reshape(1, -1)).indices
    conds = {}
    for node in path[:-1]:
        f = int(t.feature[node])
        present = bool(x[f] > t.threshold[node])
        conds.setdefault(f, present)  # binary features: first test decides
    leaf = int(path[-1])
    cluster = int(tree.classes_[np.argmax(t.value[leaf, 0])])
    clauses = [{"feature_index": f, "meaning": meanings[f], "present": p} for f, p in conds.items()]
    return {"if": clauses, "then_cluster": cluster, "text": rule_text(clauses, cluster), "length": len(clauses),
            "leaf_purity": round(float(purity[leaf]), 3), "leaf_support": int(t.n_node_samples[leaf])}


def rules_by_cluster(surrogate, meanings: list[str], top: int = 3) -> dict[int, list[dict]]:
    """All leaves as rules, grouped by the cluster they predict, strongest support first."""
    tree = surrogate.tree
    t, purity = _leaf_stats(tree)
    out = defaultdict(list)

    def walk(node, conds):
        if t.children_left[node] == -1:
            cluster = int(tree.classes_[np.argmax(t.value[node, 0])])
            seen, clauses = set(), []
            for f, p in conds:
                if f not in seen:
                    seen.add(f)
                    clauses.append({"meaning": meanings[f], "present": p})
            out[cluster].append({"text": rule_text(clauses, cluster), "length": len(clauses),
                                 "support": int(t.n_node_samples[node]),
                                 "purity": round(float(purity[node]), 3)})
            return
        f = int(t.feature[node])
        walk(t.children_left[node], conds + [(f, False)])
        walk(t.children_right[node], conds + [(f, True)])

    walk(0, [])
    return {c: sorted(v, key=lambda r: -r["support"])[:top] for c, v in out.items()}
