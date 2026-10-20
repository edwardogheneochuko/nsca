"""Rule-based symptom extraction: matches free text against the dataset's own symptom names.

Deliberately simple and transparent: no synonym table is invented. A symptom is matched
only when its dataset name (underscores read as spaces) appears as a whole phrase.
Phrases preceded within three words by a negation ("no", "not", "without", ...) are
reported as negated and are NOT used as observed symptoms.
"""
from __future__ import annotations

import re

NEGATIONS = {"no", "not", "without", "never", "denies", "denied", "neither", "nor", "cannot", "cant", "dont", "doesnt", "didnt"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower().replace("'", ""))).strip()


def extract_symptoms(text: str, feature_names: list[str]) -> tuple[list[str], list[str]]:
    """Return (matched, negated) dataset symptom names, in order of appearance."""
    t = " " + _norm(text) + " "
    consumed = [False] * len(t)
    hits: list[tuple[int, str, bool]] = []
    for name in sorted(feature_names, key=lambda n: -len(n)):  # longest phrase first
        phrase = name.replace("_", " ")
        for m in re.finditer(r"(?<= )" + re.escape(phrase) + r"(?= )", t):
            if any(consumed[m.start():m.end()]):
                continue
            for k in range(m.start(), m.end()):
                consumed[k] = True
            before = t[:m.start()].split()[-3:]
            hits.append((m.start(), name, any(w in NEGATIONS for w in before)))
    hits.sort()
    matched = list(dict.fromkeys(n for _, n, neg in hits if not neg))
    negated = list(dict.fromkeys(n for _, n, neg in hits if neg))
    return matched, negated
