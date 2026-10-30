"""FastAPI backend for the neuro-symbolic disease identification pipeline.

Run locally:   uvicorn app.api.main:app --reload
Render start:  uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import IdentifyRequest, IdentifyResponse, ParseRequest, ParseResponse, Question, Symptom
from app.config import DATASETS, DEFAULT_DATASET, MODEL_DIR
from app.engine import Engine

ENGINES: dict[str, Engine] = {}


@asynccontextmanager
async def lifespan(_app):
    """Load trained models once at startup (no training on the server)."""
    for d in DATASETS:
        if (MODEL_DIR / d / "meta.json").exists():
            ENGINES[d] = Engine(d)
    yield
    ENGINES.clear()


app = FastAPI(title="NSCA – neuro-symbolic disease identification",
              description="ESOINN → XAI/fusion → HPO symbolic reasoning, following the Pipeline B flow chart.",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


def engine(name: str | None) -> Engine:
    name = name or DEFAULT_DATASET
    if name not in ENGINES:
        raise HTTPException(503, f"model '{name}' is not loaded; train it with scripts/train.py")
    return ENGINES[name]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(ENGINES), "default_dataset": DEFAULT_DATASET}


@app.get("/model-status")
def model_status(dataset: str | None = Query(None)):
    return engine(dataset).status()


@app.get("/symptoms", response_model=list[Symptom])
def symptoms(dataset: str | None = Query(None), include_antecedents: bool = False, q: str | None = None):
    e = engine(dataset)
    out = []
    for f, m, a, (hid, lab) in zip(e.features, e.meanings, e.is_antecedent,
                                   zip(e.grounding.hpo_id, e.grounding.hpo_label)):
        if a and not include_antecedents:
            continue
        if q and q.lower() not in m.lower() and q.lower() not in str(lab).lower():
            continue
        out.append(Symptom(id=f, meaning=m, hpo_id=hid if isinstance(hid, str) else None,
                           hpo_label=lab if isinstance(lab, str) else None, kind="antecedent" if a else "symptom"))
    return out


@app.get("/questions", response_model=list[Question])
def questions(dataset: str | None = Query(None), include_risk_factors: bool = True):
    """Questionnaire for the app: each question with its type and answer options.

    Send the chosen option `value`s (and the `id` of every yes/no question answered yes) to /identify.
    Show a question with a `parent` only when the parent was answered yes.
    """
    qs = engine(dataset).questions
    return [q for q in qs if include_risk_factors or q["kind"] == "symptom"]


@app.post("/parse-text", response_model=ParseResponse)
def parse_text(req: ParseRequest):
    """Free text -> questionnaire answers, e.g. "I have itching and feeling unwell".

    Rule-based and explainable: each match says which phrase triggered it. Negated symptoms
    ("no fever") are returned separately and not ticked. Nothing is diagnosed here; send the
    returned `symptoms` plus the values in `answers` to /identify after the user confirms.
    """
    return engine(req.dataset).parser.parse(req.text)


@app.post("/identify", response_model=IdentifyResponse)
def identify(req: IdentifyRequest):
    e = engine(req.dataset)
    x, unknown, filled = e.encode(req.symptoms, req.fill_defaults)
    if x.sum() == 0:
        raise HTTPException(422, {"error": "none of the symptoms were recognised", "unknown_symptoms": unknown})
    r = e.identify(x)
    rec = r["stage2_fused_record"]
    summary = {
        "cluster": {"id": rec["cluster_id"], "name": rec["cluster_name"]},
        "rule": rec["rule"]["text"],
        "record_reliable": rec["reliable"], "unreliable_reasons": rec["unreliable_reasons"],
        "fidelity": rec["fidelity"],
        "top_candidates": [{k: c[k] for k in ("disease", "score", "consistent", "failed_checks")}
                           for c in r["stage3_symbolic"]["candidates"]],
        "reason": r["stage3_symbolic"]["resolution"]["reason"],
    }
    return IdentifyResponse(outcome=r["outcome"], disease=r["disease"], confidence=r["confidence"],
                            unknown_symptoms=unknown, filled_defaults=filled, summary=summary,
                            stages={k: r[k] for k in ("stage1_neural", "stage2_fused_record", "stage3_symbolic")}
                            if req.detail == "full" else None,
                            elapsed_ms=r["elapsed_ms"])


@app.get("/flags")
def flags(dataset: str | None = Query(None)):
    """Patients flagged as possible new disease during this server session (provisional)."""
    return engine(dataset).flagged


@app.get("/clusters")
def clusters(dataset: str | None = Query(None)):
    """Every ESOINN cluster with its post-hoc name, size and strongest surrogate rules."""
    from app.stage2_xai.rules import rules_by_cluster
    e = engine(dataset)
    rb = rules_by_cluster(e.surrogate, e.meanings)
    return [{"cluster_id": c, "name": e.esoinn.cluster_label.get(c), "training_patients": e.esoinn.cluster_size.get(c),
             "radius": e.esoinn.cluster_radius.get(c), "rules": rb.get(c, [])}
            for c in sorted(e.esoinn.cluster_size)]
