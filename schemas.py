"""Request/response models. The frontend types in nsca-mobile/types must mirror these exactly."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Outcome = Literal["known_pattern", "potentially_novel_pattern", "inconclusive"]
Severity = Literal["mild", "moderate", "severe"]


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool


class EvaluationSummary(BaseModel):
    performed: bool
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    known_pattern: Optional[int] = None
    potentially_novel_pattern: Optional[int] = None
    inconclusive: Optional[int] = None
    known_pattern_accuracy: Optional[float] = None
    coverage: Optional[float] = None
    note: str


class ModelStatusResponse(BaseModel):
    dataset_loaded: bool
    original_path: str
    processed_dir: str
    processed_written: bool
    dataset_sha256: str
    n_raw_rows: int
    n_rows_dropped: int
    n_duplicates_removed: int
    n_conflicting_patterns: int
    n_training_rows: int
    n_diseases: int
    n_symptom_features: int
    esoinn_trained: bool
    n_nodes: int
    n_clusters: int
    evaluation: Optional[EvaluationSummary] = None
    research_prototype_notice: str


class SymptomExtractRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)


class SymptomExtractResponse(BaseModel):
    matched_symptoms: list[str]
    negated_symptoms: list[str]


class SymptomListResponse(BaseModel):
    symptoms: list[str]


class IdentifyRequest(BaseModel):
    description: str = Field(min_length=1, max_length=2000)
    duration: Optional[str] = Field(default=None, max_length=100)
    severity: Optional[Severity] = None
    symptoms: Optional[list[str]] = Field(default=None, max_length=50)  # optional explicit symptom names


class Evidence(BaseModel):
    matched_symptoms: list[str]
    typical_symptoms: list[str]
    cluster_support: Optional[int] = None
    cluster_purity: Optional[float] = None
    distance: Optional[float] = None
    similarity_threshold: Optional[float] = None


class IdentifyResponse(BaseModel):
    outcome: Outcome
    reason_code: str
    disease: Optional[str] = None
    observed_symptoms: list[str]
    unrecognised_symptoms: list[str]
    cluster_id: Optional[int] = None
    node_id: Optional[int] = None
    evidence: Evidence
    explanation: str
    human_review_required: bool
    duration: Optional[str] = None
    severity: Optional[Severity] = None
    urgent_care_notice: str
    disclaimer: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
