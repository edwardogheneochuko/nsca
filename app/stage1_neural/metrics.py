"""Stage 1 evaluation: cluster quality against the hidden disease labels."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def purity(clusters, labels) -> float:
    tot = 0
    for c in np.unique(clusters):
        tot += Counter(labels[clusters == c]).most_common(1)[0][1]
    return tot / len(labels)


def stage1_report(model, X, y, n_true: int) -> dict:
    p = model.predict(X)
    cl = p["cluster"]
    ok = ~p["outlier"]
    n_nodes = len(model.nodes)
    n_clusters = len(set(model.cluster[model.nodes].tolist()))
    # how many true diseases are the majority of at least one cluster
    covered = len({model.cluster_label.get(int(c)) for c in np.unique(cl)} & set(y.tolist()))
    return {
        "nodes": n_nodes, "clusters": n_clusters, "true_diseases": n_true,
        "diseases_recovered_as_cluster_majority": covered,
        "ARI": round(float(adjusted_rand_score(y, cl)), 4),
        "NMI": round(float(normalized_mutual_info_score(y, cl)), 4),
        "purity": round(float(purity(cl, y)), 4),
        "outlier_rate": round(float((~ok).mean()), 4),
        "ARI_non_outliers": round(float(adjusted_rand_score(y[ok], cl[ok])), 4) if ok.sum() > 1 else None,
        "noise_buffer_patients": len(model.rejected_ids),
    }
