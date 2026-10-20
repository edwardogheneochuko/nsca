from pydantic import BaseModel, Field
from typing import Literal

class IdentifyRequest(BaseModel):
    description: str = Field(min_length=3, max_length=5000)
    age_years: int | None = Field(default=None, ge=0, le=120)
    symptom_duration: str | None = Field(default=None, max_length=200)
    severity: Literal["mild", "moderate", "severe", "unknown"] = "unknown"

class ExtractRequest(BaseModel):
    description: str = Field(min_length=1, max_length=5000)

class DiseaseInfo(BaseModel):
    id: str
    name: str
    terminology: str | None = None
    code: str | None = None
    evidence_note: str

class Explanation(BaseModel):
    observed_symptoms: list[str]
    supporting_evidence: list[str]
    missing_information: list[str]
    cluster_id: str | None = None
    reasoning_summary: str

class IdentifyResponse(BaseModel):
    case_id: str
    outcome: Literal["known_pattern", "potentially_novel_pattern", "inconclusive"]
    disease: DiseaseInfo | None
    explanation: Explanation
    review_required: bool
    disclaimer: str

class FeedbackRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=100)
    reviewer_outcome: Literal["confirmed_association", "not_supported", "needs_review"]
    notes: str | None = Field(default=None, max_length=3000)
