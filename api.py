
"""FastAPI app: health, model status, symptom extraction, identification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dataset_loader import DatasetError, load_dataset
from engine import NSCAEngine, evaluate_holdout
from extractor import extract_symptoms
from schemas import (
    Evidence,
    HealthResponse,
    IdentifyRequest,
    IdentifyResponse,
    ModelStatusResponse,
    SymptomExtractRequest,
    SymptomExtractResponse,
)

NOTICE = (
    "NSCA is a research prototype. It is not a clinically validated diagnostic tool "
    "and must not replace professional medical advice."
)

URGENT = (
    "If you have severe symptoms such as chest pain, difficulty breathing, sudden weakness, "
    "confusion, heavy bleeding or thoughts of self-harm, seek emergency care or call your local "
    "emergency number now. Do not wait for this tool."
)

URGENT_SEVERE = (
    "You reported severe symptoms. Please seek urgent medical care. " + URGENT
)

DEFAULT_ORIGINS = [
    "http://localhost:8081",
    "http://localhost:19006",
    "http://127.0.0.1:8081",
    "http://127.0.0.1:19006",
]


def cors_origins() -> list[str]:
    extra = os.getenv("NSCA_CORS_ORIGINS", "")
    origins = DEFAULT_ORIGINS + [
        origin.strip()
        for origin in extra.split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys(origins))


def create_app(
    original_path: str | None = None,
    processed_dir: str | None = None,
    run_evaluation: bool = True,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.dataset = None
        app.state.engine = None
        app.state.error = None

        try:
            ds = load_dataset(
                original_path or os.getenv("NSCA_DATASET_PATH"),
                processed_dir or os.getenv("NSCA_PROCESSED_DIR"),
            )

            app.state.dataset = ds

            engine = NSCAEngine(ds)

            if run_evaluation:
                engine.evaluation = evaluate_holdout(ds)

            app.state.engine = engine

        except DatasetError as exc:
            app.state.error = str(exc)

        except Exception as exc:
            # Training failure must not crash the server.
            app.state.error = f"Model training failed: {exc}"

        yield

    app = FastAPI(
        title="NSCA Backend",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error.",
                "error_code": "internal",
            },
        )

    def require_engine(request: Request) -> NSCAEngine:
        eng = request.app.state.engine

        if eng is None:
            raise HTTPException(
                503,
                request.app.state.error or "Model not ready.",
            )

        return eng

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        return HealthResponse(
            status="ok",
            dataset_loaded=request.app.state.dataset is not None,
        )

    @app.get("/model/status", response_model=ModelStatusResponse)
    def model_status(request: Request):
        eng = require_engine(request)
        ds, st = eng.dataset, eng.status()

        return ModelStatusResponse(
            dataset_loaded=True,
            original_path=ds.original_path,
            processed_dir=ds.processed_dir,
            processed_written=ds.processed_written,
            dataset_sha256=ds.sha256,
            n_raw_rows=ds.n_raw_rows,
            n_rows_dropped=ds.n_rows_dropped,
            n_duplicates_removed=ds.n_duplicates_removed,
            n_conflicting_patterns=ds.n_conflicting_patterns,
            n_training_rows=ds.n_rows,
            n_diseases=ds.n_diseases,
            n_symptom_features=ds.n_symptoms,
            esoinn_trained=st["esoinn_trained"],
            n_nodes=st["n_nodes"],
            n_clusters=st["n_clusters"],
            evaluation=st["evaluation"],
            research_prototype_notice=NOTICE,
        )

    @app.post(
        "/symptoms/extract",
        response_model=SymptomExtractResponse,
    )
    def extract(body: SymptomExtractRequest, request: Request):
        eng = require_engine(request)

        matched, negated = extract_symptoms(
            body.description,
            eng.dataset.symptoms,
        )

        return SymptomExtractResponse(
            matched_symptoms=matched,
            negated_symptoms=negated,
        )

    @app.post("/identify", response_model=IdentifyResponse)
    def identify(body: IdentifyRequest, request: Request):
        eng = require_engine(request)

        matched, _ = extract_symptoms(
            body.description,
            eng.dataset.symptoms,
        )

        symptoms = list(
            dict.fromkeys([
                *matched,
                *(body.symptoms or []),
            ])
        )

        r = eng.identify(symptoms)

        return IdentifyResponse(
            outcome=r.outcome,
            reason_code=r.reason_code,
            disease=r.disease,
            observed_symptoms=r.observed_symptoms,
            unrecognised_symptoms=r.unrecognised_symptoms,
            cluster_id=r.cluster_id,
            node_id=r.node_id,
            evidence=Evidence(
                matched_symptoms=r.matched_symptoms,
                typical_symptoms=r.typical_symptoms,
                cluster_support=r.cluster_support,
                cluster_purity=r.cluster_purity,
                distance=r.distance,
                similarity_threshold=r.similarity_threshold,
            ),
            explanation=r.explanation,
            human_review_required=r.human_review_required,
            duration=body.duration,
            severity=body.severity,
            urgent_care_notice=(
                URGENT_SEVERE
                if body.severity == "severe"
                else URGENT
            ),
            disclaimer=NOTICE,
        )

    return app


app = create_app()
