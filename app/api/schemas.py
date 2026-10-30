from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IdentifyRequest(BaseModel):
    symptoms: list[str] = Field(..., min_length=1,
                                description="Symptom ids (e.g. 'E_91'), plain names ('fever') or HPO labels")
    dataset: Literal["mendeley", "ddxplus", "phenopackets"] | None = None
    detail: Literal["summary", "full"] = "full"
    fill_defaults: bool = Field(True, description="DDXPlus: fill default answers for unanswered follow-ups and travel")


class IdentifyResponse(BaseModel):
    outcome: Literal["known_disease", "flag_as_new"]
    disease: str | None
    confidence: str
    unknown_symptoms: list[str]
    filled_defaults: list[str] = []
    summary: dict[str, Any]
    stages: dict[str, Any] | None = None
    elapsed_ms: float


class Symptom(BaseModel):
    id: str
    meaning: str
    hpo_id: str | None
    hpo_label: str | None
    kind: Literal["symptom", "antecedent"]


class Option(BaseModel):
    value: str
    label: str


class Question(BaseModel):
    id: str
    text: str
    type: Literal["yes_no", "single", "multi", "scale"]
    kind: Literal["symptom", "risk_factor"]
    parent: str | None
    options: list[Option]
    default: str | None


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description='e.g. "I have itching and feeling unwell"')
    dataset: Literal["mendeley", "ddxplus", "phenopackets"] | None = None


class ParsedItem(BaseModel):
    phrase: str
    concept: str
    question_id: str
    question: str
    source: str
    note: str | None


class ParseResponse(BaseModel):
    symptoms: list[str]                 # question ids to tick (send these to /identify)
    answers: dict[str, list[str]]       # follow-up answers found (e.g. pain location)
    found: list[ParsedItem]
    absent: list[ParsedItem]            # mentioned with a negation ("no fever")
    unrecognised: list[str]             # parts of the text that matched nothing
