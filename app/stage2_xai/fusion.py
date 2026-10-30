"""Stage 2 – Fused explanation record ("rules, evidence, exemplars, fidelity").

The record is the only thing Stage 3 reads.  It is marked unreliable when CART
and ESOINN disagree, when surrogate fidelity is low, or when Stage 1 found the
patient outside the winner cluster's radius.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.stage2_xai.surrogate import FIDELITY_THRESHOLD


class FusedRecord(BaseModel):
    cluster_id: int
    cluster_name: str | None          # post-hoc majority disease of the cluster (from training labels)
    runner_up_cluster: int | None
    runner_up_name: str | None
    winner_node: int
    distance_to_winner: float
    cluster_radius: float
    stage1_outlier: bool
    rule: dict[str, Any]              # surrogate if-then rule for this patient
    cart_cluster: int
    agreement: bool                   # CART prediction == ESOINN cluster
    fidelity: float                   # surrogate agreement with ESOINN on held-out patients
    node_difference: dict[str, Any]
    exemplars: list[dict[str, Any]]
    reliable: bool
    unreliable_reasons: list[str]


def fuse(*, cluster_id, cluster_name, runner_up_cluster, runner_up_name, winner_node, dist, radius,
         outlier, rule, cart_cluster, fidelity, node_diff, exemplars) -> FusedRecord:
    reasons = []
    agreement = int(cart_cluster) == int(cluster_id)
    if not agreement:
        reasons.append(f"CART predicts cluster {cart_cluster}, ESOINN says {cluster_id}")
    if fidelity < FIDELITY_THRESHOLD:
        reasons.append(f"surrogate fidelity {fidelity:.2f} below {FIDELITY_THRESHOLD}")
    if outlier:
        reasons.append("patient lies outside the winner cluster's radius")
    return FusedRecord(cluster_id=int(cluster_id), cluster_name=cluster_name,
                       runner_up_cluster=None if runner_up_cluster is None or runner_up_cluster < 0 else int(runner_up_cluster),
                       runner_up_name=runner_up_name, winner_node=int(winner_node),
                       distance_to_winner=round(float(dist), 4), cluster_radius=round(float(radius), 4),
                       stage1_outlier=bool(outlier), rule=rule, cart_cluster=int(cart_cluster), agreement=agreement,
                       fidelity=round(float(fidelity), 4), node_difference=node_diff, exemplars=exemplars,
                       reliable=not reasons, unreliable_reasons=reasons)
