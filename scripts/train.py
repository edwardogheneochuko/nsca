"""Offline training: build every stage for one dataset and save the model state.

    python -m scripts.train --dataset ddxplus
    python -m scripts.train --dataset mendeley

Training happens here, never on the server; the API only loads models/<dataset>/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.config import MODEL_DIR
from app.data.loaders import load
from app.stage1_neural.esoinn import ESOINN
from app.stage1_neural.metrics import stage1_report
from app.stage2_xai.exemplars import Exemplars
from app.stage2_xai.surrogate import CARTSurrogate
from app.stage3_symbolic import candidates as cand_mod
from app.stage3_symbolic.consistency import calibrate
from app.stage3_symbolic.grounding import build_grounding
from app.stage3_symbolic.hpo import get_hpo
from app.stage3_symbolic.knowledge_base import KnowledgeBase, TermSpace

PARAMS = {"lam": 300, "age_max": 5, "t_max": 0.7, "c1": 0.001, "c2": 1.0}
# Real case reports are sparser and more varied: fewer prunings (larger lambda) and a tighter cap work better.
PARAMS_BY_DATASET = {"phenopackets": {"lam": 600, "age_max": 5, "t_max": 0.6, "c1": 0.001, "c2": 1.0}}


def train(name: str, seed_order: int | None = None, log=print) -> dict:
    t0 = time.time()
    d = load(name)
    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    order = d.stream_order if seed_order is None else np.random.default_rng(seed_order).permutation(d.stream_order)

    log(f"[{name}] Stage 1: ESOINN on {len(order)} streamed patients, {d.X_train.shape[1]} features")
    params = PARAMS_BY_DATASET.get(name, PARAMS)
    esoinn = ESOINN(d.X_train.shape[1], **params).fit(d.X_train, order)
    esoinn.finalize(d.X_train, d.y_train)
    esoinn.save(out / "esoinn")
    s1 = stage1_report(esoinn, d.X_val, d.y_val, len(d.known_diseases))
    log(f"[{name}]   {s1}")

    log(f"[{name}] Stage 2: surrogate, exemplars")
    ptr, pva = esoinn.predict(d.X_train), esoinn.predict(d.X_val)
    sur = CARTSurrogate().fit(d.X_train, ptr["cluster"], d.X_val, pva["cluster"], esoinn.structure_version)
    sur.save(out)
    ex = Exemplars().fit(d.X_train, ptr["cluster"], d.y_train)
    (out / "exemplars.json").write_text(json.dumps({str(c): {"train_index": v["train_index"], "X": v["X"].tolist(),
                                                              "labels": v["labels"]} for c, v in ex.by_cluster.items()}))
    log(f"[{name}]   fidelity={sur.fidelity:.4f} leaves={sur.tree.get_n_leaves()}")

    log(f"[{name}] Stage 3: HPO grounding, knowledge base, calibration")
    hpo = get_hpo()
    g = build_grounding(name, d.features, d.meanings, d.is_antecedent, hpo)
    g.to_csv(out / "grounding.csv", index=False)
    space = TermSpace(g, d.is_antecedent)
    kb = KnowledgeBase.build(space.to_terms(d.X_train), d.y_train, space, d.X_train, hpo, d.meanings)
    kb.save(out / "knowledge_base.json")
    R = cand_mod.related_matrix(kb, hpo)
    th = calibrate(space.to_terms(d.X_val), d.y_val, kb, R)
    log(f"[{name}]   grounded methods={g.method.value_counts().to_dict()} thresholds={th.to_dict()}")

    meta = {"dataset": name, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "params": params,
            "features": d.features, "meanings": d.meanings, "is_antecedent": d.is_antecedent.tolist(),
            "known_diseases": d.known_diseases, "heldout_diseases": d.heldout_diseases,
            "thresholds": th.to_dict(), "top_k": 3, "hpo_release": hpo.data_version,
            "stage1_validation": s1, "surrogate_fidelity": sur.fidelity, "source": d.info.get("source"),
            "train_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"[{name}] saved to {out} in {meta['train_seconds']} s")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendeley", "ddxplus", "phenopackets", "all"], default="all")
    a = ap.parse_args()
    for n in (["mendeley", "ddxplus", "phenopackets"] if a.dataset == "all" else [a.dataset]):
        train(n)
