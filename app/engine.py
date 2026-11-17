"""Engine: runs one patient through the three stages exactly as drawn in the flow chart.

Stage 1  Symptom data -> ESOINN -> Model state
Stage 2  CART surrogate -> Rule extraction ┐
         Node difference                   ├-> Fused explanation record
         Exemplars                         ┘
Stage 3  Grounding (HPO) -> Candidates -> Consistency -> Resolution -> Known disease | Flag as new
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app import questions as q_mod
from app.config import FLAG_DIR, MODEL_DIR
from app.flag_store import FlagStore, case_from_result
from app.stage1_neural.esoinn import ESOINN
from app.stage2_xai import exemplars as ex_mod
from app.stage2_xai import node_difference, rules
from app.stage2_xai.fusion import fuse
from app.stage2_xai.surrogate import CARTSurrogate
from app.stage3_symbolic import candidates as cand_mod
from app.stage3_symbolic import consistency as cons_mod
from app.stage3_symbolic.hpo import get_hpo
from app.stage3_symbolic.knowledge_base import KnowledgeBase, TermSpace
from app.stage3_symbolic.resolution import resolve


class Engine:
    def __init__(self, dataset: str, model_dir: Path | None = None):
        self.dataset = dataset
        self.dir = Path(model_dir or MODEL_DIR / dataset)
        meta = json.loads((self.dir / "meta.json").read_text())
        self.meta = meta
        self.features: list[str] = meta["features"]
        self.meanings: list[str] = meta["meanings"]
        self.is_antecedent = np.array(meta["is_antecedent"], bool)
        self.esoinn = ESOINN.load(self.dir / "esoinn")
        self.surrogate = CARTSurrogate.load(self.dir)
        self.exemplars = ex_mod.Exemplars()
        exj = json.loads((self.dir / "exemplars.json").read_text())
        self.exemplars.by_cluster = {int(c): {"train_index": v["train_index"], "X": np.array(v["X"], np.uint8),
                                              "labels": v["labels"]} for c, v in exj.items()}
        self.grounding = pd.read_csv(self.dir / "grounding.csv")
        self.space = TermSpace(self.grounding, self.is_antecedent)
        self.kb = KnowledgeBase.load(self.dir / "knowledge_base.json")
        self.hpo = get_hpo()
        self.R = cand_mod.related_matrix(self.kb, self.hpo)
        self.thresholds = cons_mod.Thresholds(**meta["thresholds"])
        self.k = meta.get("top_k", 3)
        self._lookup = {}
        for i, (f, m) in enumerate(zip(self.features, self.meanings)):
            self._lookup[f.lower()] = i
            self._lookup.setdefault(m.lower(), i)
        for i, lab in enumerate(self.grounding.hpo_label):
            if isinstance(lab, str):
                self._lookup.setdefault(lab.lower(), i)
        self.flags = FlagStore(FLAG_DIR / f"{dataset}.json", dataset)
        self.questions = q_mod.build(self.features, self.meanings, self.is_antecedent, dataset)
        self._parser = None

    @property
    def parser(self):
        """Free-text symptom parser (built on first use)."""
        if self._parser is None:
            from app.text_parse import Parser
            self._parser = Parser.build(self)
        return self._parser

    # ---------------------------------------------------------------- input
    def encode(self, symptoms: list[str], fill_defaults: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
        """Symptoms -> 0/1 vector.  Returns (vector, unrecognised inputs, default answers filled in)."""
        x = np.zeros(len(self.features), np.uint8)
        unknown = []
        for s in symptoms:
            i = self._lookup.get(str(s).strip().lower())
            if i is None:
                unknown.append(s)
            else:
                x[i] = 1
        filled = q_mod.complete_defaults(x, self.questions, self.features) if (fill_defaults and x.any()) else []
        return x, unknown, filled

    # -------------------------------------------------------------- pipeline
    def identify(self, x: np.ndarray, record_flag: bool = True, context: dict | None = None) -> dict:
        """Run all three stages. A patient resolved to `flag_as_new` is saved to the flag store
        (unless record_flag is False, as in evaluation). `context` adds the raw input to the saved case."""
        t0 = time.perf_counter()
        x = np.asarray(x, np.uint8)
        e = self.esoinn
        # Stage 1
        p = e.predict(x.reshape(1, -1))
        s1, d1 = int(p["winner"][0]), float(p["dist"][0])
        cluster = int(p["cluster"][0])
        runner, _ = e.runner_up_other_cluster(x, s1)
        stage1 = {"winner_node": s1, "cluster_id": cluster, "cluster_name": e.cluster_label.get(cluster),
                  "distance": round(d1, 4), "cluster_radius": round(e.cluster_radius.get(cluster, float("inf")), 4),
                  "outlier": bool(p["outlier"][0]), "runner_up_node": runner,
                  "runner_up_cluster": int(e.cluster[runner]) if runner >= 0 else None,
                  "features_total": int(len(x)), "features_present": int(x.sum()),
                  "network_nodes": int(len(e.nodes)), "network_clusters": int(len(set(e.cluster[e.nodes].tolist())))}
        # Stage 2
        rule = rules.rule_for_patient(self.surrogate, x, self.meanings)
        nd = node_difference.explain(x, e.W[s1], e.W[runner] if runner >= 0 else None, self.meanings)
        exs = self.exemplars.nearest(x, cluster, self.meanings)
        rc = stage1["runner_up_cluster"]
        record = fuse(cluster_id=cluster, cluster_name=stage1["cluster_name"], runner_up_cluster=rc,
                      runner_up_name=e.cluster_label.get(rc) if rc is not None else None, winner_node=s1, dist=d1,
                      radius=stage1["cluster_radius"], outlier=stage1["outlier"], rule=rule,
                      cart_cluster=rule["then_cluster"], fidelity=self.surrogate.fidelity, node_diff=nd, exemplars=exs)
        # Stage 3
        t = self.space.to_terms(x.reshape(1, -1))[0]
        grounded = [{"term": self.kb.terms[j], "label": self.kb.label(self.kb.terms[j])} for j in np.flatnonzero(t)]
        cands = cand_mod.top_k(t, self.kb, self.R, self.k, hint=record.cluster_name if record.reliable else None)
        sc = cand_mod.score_all(t, self.kb, self.R)
        sup_terms = [self.space.feature_term[s["feature_index"]] for s in nd["supporting"]
                     if self.space.feature_term[s["feature_index"]] >= 0]
        checked = [cons_mod.check(t, c, self.kb, sc["Fp"], self.thresholds, sup_terms) for c in cands]
        res = resolve(checked, record)
        out = {"dataset": self.dataset,
               "stage1_neural": stage1,
               "stage2_fused_record": record.model_dump(),
               "stage3_symbolic": {"grounding": grounded, "candidates": checked, "resolution": res,
                                   "ontology": "Human Phenotype Ontology (HPO)", "hpo_release": self.hpo.data_version},
               "outcome": res["outcome"], "disease": res["disease"], "confidence": res["confidence"],
               "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        out["flag"] = None
        if record_flag and res["outcome"] == "flag_as_new":
            ctx = dict(context or {})
            ctx["model"] = {"trained_at": self.meta.get("trained_at"), "hpo_release": self.hpo.data_version}
            out["flag"] = self.flags.save(np.flatnonzero(x).tolist(), case_from_result(out, x, self.meanings, ctx))
        return out

    def status(self) -> dict:
        s = self.esoinn.summary()
        return {"dataset": self.dataset, "trained_at": self.meta.get("trained_at"),
                "stage1_esoinn": s,
                "stage2_surrogate": {"fidelity": self.surrogate.fidelity, "leaves": int(self.surrogate.tree.get_n_leaves()),
                                     "refit_triggers_pending": self.surrogate.refit_triggers(self.esoinn)},
                "stage3": {"hpo_release": self.hpo.data_version, "known_diseases": len(self.kb.diseases),
                           "grounded_symptoms": int((self.grounding.method.isin(["exact", "curated", "override"])).sum()),
                           "symptom_columns": int((~self.is_antecedent).sum()),
                           "thresholds": self.thresholds.to_dict(), "top_k": self.k},
                "flagged_cases": self.flags.counts()}
