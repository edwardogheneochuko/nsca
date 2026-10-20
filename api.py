from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
import json

from schemas import IdentifyRequest, IdentifyResponse, ExtractRequest, FeedbackRequest
from engine import identify, extract_symptoms, KNOWN_CONCEPTS, MODEL
from storage import save_feedback, save_case, load_feedback_count

app = FastAPI(
    title="NSCA Disease Identification API",
    version="0.1.0",
    description="Research prototype. Not a medical diagnosis."
)


origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = "Research tool only; this output is not a medical diagnosis and does not replace professional medical care."

@app.get("/health")
def health():
    return {"status": "ok", "service": "nsca-disease-identification-api"}

@app.get("/api/v1/model/status")
def model_status():
    return {
        "algorithm": "ESOINN (prototype implementation)",
        "model_ready": MODEL.ready,
        "node_count": len(MODEL.nodes),
        "edge_count": len(MODEL.edges),
        "feedback_records": load_feedback_count(),
        "note": "Clusters are exploratory patterns, not clinically validated diseases."
    }

@app.post("/api/v1/symptoms/extract")
def symptoms_extract(payload: ExtractRequest):
    symptoms = extract_symptoms(payload.description)
    return {"symptoms": symptoms, "count": len(symptoms)}

@app.post("/api/v1/identify", response_model=IdentifyResponse)
def identify_case(payload: IdentifyRequest):
    symptoms = extract_symptoms(payload.description)
    if not symptoms:
        result = {
            "case_id": str(uuid4()),
            "outcome": "inconclusive",
            "disease": None,
            "explanation": {
                "observed_symptoms": [],
                "supporting_evidence": [],
                "missing_information": ["Describe symptoms in more detail, including duration and severity."],
                "cluster_id": None,
                "reasoning_summary": "No recognized symptom terms could be extracted from the description."
            },
            "review_required": True,
            "disclaimer": DISCLAIMER
        }
    else:
        result = identify(symptoms, payload.description)
        result["case_id"] = str(uuid4())
        result["disclaimer"] = DISCLAIMER

    save_case(result)
    return result

@app.get("/api/v1/knowledge/symptoms")
def knowledge_symptoms(q: str | None = None):
    values = [{"id": key, "label": value} for key, value in KNOWN_CONCEPTS.items()]
    if q:
        ql = q.lower()
        values = [item for item in values if ql in item["id"].lower() or ql in item["label"].lower()]
    return {"items": values, "source": "Built-in demonstration vocabulary; not a licensed terminology release."}

@app.get("/api/v1/knowledge/diseases/{disease_id}")
def knowledge_disease(disease_id: str):
    from engine import DEMO_DISEASES
    disease = DEMO_DISEASES.get(disease_id)
    if not disease:
        raise HTTPException(status_code=404, detail="Disease concept not found in the demonstration knowledge base.")
    return disease

@app.post("/api/v1/feedback")
def feedback(payload: FeedbackRequest):
    feedback_id = save_feedback(payload.model_dump())
    return {"feedback_id": feedback_id, "saved": True, "note": "Feedback is stored for review and is not used to automatically retrain the model."}
