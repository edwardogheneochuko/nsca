"""
Research prototype components:
- ESOINN-style incremental graph clustering for numeric symptom vectors.
- Small demonstration symptom vocabulary and explicit demonstration rules.
This is NOT a clinically validated model or production ESOINN package.
Do not use its outputs for medical decisions.
"""
from collections import defaultdict
from math import sqrt
import re
import numpy as np
import os

KNOWN_CONCEPTS = {
    "fever": "Fever",
    "headache": "Headache",
    "cough": "Cough",
    "sore_throat": "Sore throat",
    "runny_nose": "Runny nose",
    "body_ache": "Body aches",
    "fatigue": "Fatigue",
    "vomiting": "Vomiting",
    "diarrhea": "Diarrhea",
    "rash": "Rash",
    "joint_pain": "Joint pain",
    "chills": "Chills",
    "shortness_of_breath": "Shortness of breath",
    "chest_pain": "Chest pain",
    "abdominal_pain": "Abdominal pain",
    "weakness": "Weakness",
}

ALIASES = {
    "fever": ["fever", "high temperature", "hot body", "temperature"],
    "headache": ["headache", "head pain", "pain in my head"],
    "cough": ["cough", "coughing"],
    "sore_throat": ["sore throat", "throat pain"],
    "runny_nose": ["runny nose", "running nose", "nasal discharge"],
    "body_ache": ["body ache", "body aches", "muscle pain", "muscle aches"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhausted"],
    "vomiting": ["vomiting", "vomit", "throwing up"],
    "diarrhea": ["diarrhea", "diarrhoea", "loose stool"],
    "rash": ["rash", "skin rash"],
    "joint_pain": ["joint pain", "painful joints"],
    "chills": ["chills", "shivering"],
    "shortness_of_breath": ["shortness of breath", "difficulty breathing", "breathlessness"],
    "chest_pain": ["chest pain", "pain in my chest"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain"],
    "weakness": ["weakness", "weak"],
}

# Demonstration-only rules. These are intentionally not represented as a validated medical knowledge base.
DEMO_DISEASES = {
    "demo_viral_syndrome": {
        "id": "demo_viral_syndrome",
        "name": "Unspecified viral-like syndrome (demonstration concept)",
        "terminology": None,
        "code": None,
        "evidence_note": "Demonstration rule only; not a validated disease association and not mapped to SNOMED CT or ICD-11."
    }
}
DEMO_RULES = {
    "demo_viral_syndrome": {"required_any": {"fever", "cough", "sore_throat", "runny_nose", "body_ache"}, "min_matches": 3}
}

def extract_symptoms(text: str) -> list[str]:
    lower = re.sub(r"[^a-z0-9\s'-]", " ", text.lower())
    found = set()
    for symptom, phrases in ALIASES.items():
        for phrase in phrases:
            if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", lower):
                found.add(symptom)
                break
    # Simple negation filter avoids counting a directly negated phrase such as "no fever".
    for symptom, phrases in ALIASES.items():
        for phrase in phrases:
            if re.search(r"\b(no|not|without|denies|don't have|do not have)\s+(?:any\s+)?" + re.escape(phrase) + r"\b", lower):
                found.discard(symptom)
    return sorted(found)

class ESOINNPrototype:
    """
    Compact incremental graph-clustering prototype inspired by ESOINN.
    A production implementation should be validated against the ESOINN paper/reference
    implementation and evaluated on representative clinical data.
    """
    def __init__(self, dimension: int, similarity_threshold: float = 0.45):
        self.dimension = dimension
        self.threshold = similarity_threshold
        self.nodes: list[np.ndarray] = []
        self.counts: list[int] = []
        self.edges: set[tuple[int, int]] = set()
        self.ready = False

    def fit_one(self, vector: np.ndarray):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (self.dimension,):
            raise ValueError("Unexpected vector dimension")
        if not self.nodes:
            self.nodes.append(vector.copy())
            self.counts.append(1)
            self.ready = True
            return 0
        distances = [float(np.linalg.norm(vector - node)) for node in self.nodes]
        order = np.argsort(distances)
        nearest, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
        if distances[nearest] > self.threshold:
            self.nodes.append(vector.copy())
            self.counts.append(1)
            new_index = len(self.nodes) - 1
            if len(order):
                self.edges.add(tuple(sorted((new_index, nearest))))
            return new_index
        n = self.counts[nearest]
        self.nodes[nearest] = self.nodes[nearest] + (vector - self.nodes[nearest]) / (n + 1)
        self.counts[nearest] += 1
        if second != nearest and distances[second] <= self.threshold * 1.5:
            self.edges.add(tuple(sorted((nearest, second))))
        return nearest

    def nearest(self, vector: np.ndarray):
        if not self.nodes:
            return None, None
        distances = [float(np.linalg.norm(vector - node)) for node in self.nodes]
        idx = int(np.argmin(distances))
        return idx, distances[idx]

    def cluster_id(self, node_id: int) -> str:
        # Prototype reports node IDs, not a clinically meaningful disease cluster.
        return f"esoinn-node-{node_id}"

SYMPTOM_ORDER = list(KNOWN_CONCEPTS.keys())
MODEL = ESOINNPrototype(len(SYMPTOM_ORDER))

def encode(symptoms: list[str]) -> np.ndarray:
    selected = set(symptoms)
    return np.array([1.0 if s in selected else 0.0 for s in SYMPTOM_ORDER], dtype=float)

def identify(symptoms: list[str], description: str) -> dict:
    vector = encode(symptoms)
    node_id = MODEL.fit_one(vector)
    nearest_id, distance = MODEL.nearest(vector)
    missing = []
    if len(symptoms) < 2:
        missing.append("More symptom detail is needed before an association can be assessed.")
    if not any(s in symptoms for s in ("fever", "cough", "vomiting", "diarrhea", "rash", "shortness_of_breath", "chest_pain", "abdominal_pain")):
        missing.append("Add relevant symptoms and their duration, severity, and progression.")
    urgent = {"shortness_of_breath", "chest_pain"}
    if urgent.intersection(symptoms):
        return {
            "outcome": "inconclusive",
            "disease": None,
            "explanation": {
                "observed_symptoms": [KNOWN_CONCEPTS[s] for s in symptoms],
                "supporting_evidence": [],
                "missing_information": ["Urgent symptoms require prompt assessment by a qualified medical professional."],
                "cluster_id": MODEL.cluster_id(node_id),
                "reasoning_summary": "The system does not attempt to diagnose potentially urgent symptoms. Seek urgent medical care, especially for severe or sudden symptoms."
            },
            "review_required": True
        }

    matches = []
    for disease_id, rule in DEMO_RULES.items():
        count = len(set(symptoms) & rule["required_any"])
        if count >= rule["min_matches"]:
            matches.append((disease_id, count))

    if len(symptoms) < 2:
        outcome, disease, support, summary, review = "inconclusive", None, [], "The available symptom information is insufficient for even a preliminary pattern association.", True
    elif matches:
        disease_id, count = matches[0]
        outcome = "known_pattern"
        disease = DEMO_DISEASES[disease_id]
        support = [f"Demonstration rule matched {count} symptoms associated with a broad viral-like pattern."]
        summary = "A demonstration rule matched this symptom pattern. This is not a validated clinical inference; the prototype knowledge base must be replaced with reviewed medical data."
        review = True
    elif distance is not None and distance > MODEL.threshold:
        outcome, disease, support = "potentially_novel_pattern", None, []
        summary = "The input formed a relatively distant pattern in this prototype's feature space. This is only a research flag and does not establish a new disease."
        review = True
    else:
        outcome, disease, support = "inconclusive", None, []
        summary = "The current demonstration knowledge base does not provide enough validated evidence to support a disease association."
        review = True

    return {
        "outcome": outcome,
        "disease": disease,
        "explanation": {
            "observed_symptoms": [KNOWN_CONCEPTS[s] for s in symptoms],
            "supporting_evidence": support,
            "missing_information": missing,
            "cluster_id": MODEL.cluster_id(node_id),
            "reasoning_summary": summary
        },
        "review_required": review
    }
