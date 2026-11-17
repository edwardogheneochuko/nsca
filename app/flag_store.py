"""Saved "possible new disease" cases.

When Stage 3 resolves a patient to `flag_as_new`, the case is written to disk so it survives a server
restart and can be reviewed later. One JSON file per dataset (default `flags/<dataset>.json`; set
`NSCA_FLAG_DIR` to move it). The file is plain JSON so it can be opened, copied into a report or
analysed without the server.

Each saved case keeps everything needed to review it: what the patient entered, the HPO terms, the
Stage 1 cluster and outlier distance, the Stage 2 reliability, every Stage 3 candidate with the checks
it failed, and links to earlier saved cases with a similar symptom pattern. Several similar flagged
cases are the first evidence that a genuinely new disease may be appearing; one case alone is not.

Review status: pending_review -> confirmed_new | known_disease | dismissed.
"""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

REVIEW_STATUSES = ("pending_review", "confirmed_new", "known_disease", "dismissed")
SIMILAR_MAX_DISTANCE = 0.5      # Jaccard distance: share at least half of their symptoms
DUPLICATE_WINDOW_S = 600        # same symptoms re-submitted within 10 minutes -> same case, not a new one


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


class FlagStore:
    def __init__(self, path: Path, dataset: str):
        self.path = Path(path)
        self.dataset = dataset
        self._lock = threading.Lock()
        self.cases: list[dict] = []
        if self.path.exists():
            try:
                self.cases = json.loads(self.path.read_text(encoding="utf-8")).get("cases", [])
            except (json.JSONDecodeError, OSError):
                # keep the damaged file for inspection and start a fresh one
                self.path.rename(self.path.with_suffix(f".corrupt-{int(time.time())}.json"))
                self.cases = []

    # ------------------------------------------------------------ storage
    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"dataset": self.dataset, "cases": self.cases}, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written file

    def _next_id(self) -> str:
        n = max((int(c["case_id"].rsplit("-", 1)[1]) for c in self.cases), default=0) + 1
        return f"NEW-{self.dataset[:3].upper()}-{n:04d}"

    # ------------------------------------------------------------ public
    def save(self, features: list[int], case: dict) -> dict:
        """Save a flagged case. Returns {case_id, saved, duplicate_of_recent, similar_cases}."""
        fset = set(int(i) for i in features)
        now = time.time()
        with self._lock:
            for c in reversed(self.cases):
                if now - c["created_unix"] > DUPLICATE_WINDOW_S:
                    break
                if set(c["feature_indices"]) == fset:
                    return {"case_id": c["case_id"], "saved": False, "duplicate_of_recent": True,
                            "similar_cases": c["similar_cases"]}
            similar = sorted(
                ({"case_id": c["case_id"], "distance": round(jaccard(fset, set(c["feature_indices"])), 3),
                  "status": c["review"]["status"]} for c in self.cases
                 if c["review"]["status"] != "dismissed"),
                key=lambda s: s["distance"])
            similar = [s for s in similar if s["distance"] <= SIMILAR_MAX_DISTANCE][:10]
            cid = self._next_id()
            rec = {"case_id": cid, "dataset": self.dataset,
                   "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "created_unix": now,
                   "feature_indices": sorted(fset), **case, "similar_cases": similar,
                   "review": {"status": "pending_review", "note": None, "disease": None, "reviewed_at": None}}
            self.cases.append(rec)
            # link back so older cases also list the new similar one
            for s in similar:
                old = self.get(s["case_id"])
                if old is not None:
                    old["similar_cases"] = sorted(old["similar_cases"] + [
                        {"case_id": cid, "distance": s["distance"], "status": "pending_review"}],
                        key=lambda t: t["distance"])[:10]
            self._write()
        return {"case_id": cid, "saved": True, "duplicate_of_recent": False, "similar_cases": similar}

    def get(self, case_id: str) -> dict | None:
        return next((c for c in self.cases if c["case_id"] == case_id), None)

    def list(self, status: str | None = None) -> list[dict]:
        out = [c for c in self.cases if status is None or c["review"]["status"] == status]
        return list(reversed(out))  # newest first

    def review(self, case_id: str, status: str, note: str | None = None, disease: str | None = None) -> dict:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"status must be one of {REVIEW_STATUSES}")
        with self._lock:
            c = self.get(case_id)
            if c is None:
                raise KeyError(case_id)
            c["review"] = {"status": status, "note": note, "disease": disease,
                           "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            for other in self.cases:  # keep the status shown in similar-case links current
                for s in other["similar_cases"]:
                    if s["case_id"] == case_id:
                        s["status"] = status
            self._write()
            return c

    def delete(self, case_id: str) -> bool:
        with self._lock:
            n = len(self.cases)
            self.cases = [c for c in self.cases if c["case_id"] != case_id]
            for other in self.cases:
                other["similar_cases"] = [s for s in other["similar_cases"] if s["case_id"] != case_id]
            if len(self.cases) != n:
                self._write()
            return len(self.cases) != n

    def clear(self, status: str | None = None) -> int:
        """Delete every case (or every case with this review status). Returns how many were deleted."""
        with self._lock:
            gone = {c["case_id"] for c in self.cases if status is None or c["review"]["status"] == status}
            self.cases = [c for c in self.cases if c["case_id"] not in gone]
            for other in self.cases:
                other["similar_cases"] = [s for s in other["similar_cases"] if s["case_id"] not in gone]
            if gone:
                self._write()
            return len(gone)

    def counts(self) -> dict:
        out = {s: 0 for s in REVIEW_STATUSES}
        for c in self.cases:
            out[c["review"]["status"]] += 1
        out["total"] = len(self.cases)
        return out

    def groups(self) -> list[list[str]]:
        """Connected groups of similar, non-dismissed cases (size >= 2): candidate new diseases."""
        ids = [c["case_id"] for c in self.cases if c["review"]["status"] != "dismissed"]
        parent = {i: i for i in ids}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for c in self.cases:
            if c["case_id"] not in parent:
                continue
            for s in c["similar_cases"]:
                if s["case_id"] in parent:
                    parent[find(s["case_id"])] = find(c["case_id"])
        g: dict[str, list[str]] = {}
        for i in ids:
            g.setdefault(find(i), []).append(i)
        return sorted((v for v in g.values() if len(v) > 1), key=len, reverse=True)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["case_id", "created_at", "review_status", "review_disease", "review_note", "symptoms",
                    "hpo_terms", "free_text", "stage1_cluster", "stage1_distance", "stage1_outlier",
                    "stage2_reliable", "nearest_known", "candidates_failed_checks", "similar_cases"])
        for c in self.cases:
            w.writerow([c["case_id"], c["created_at"], c["review"]["status"], c["review"]["disease"] or "",
                        c["review"]["note"] or "", "; ".join(c["symptoms"]),
                        "; ".join(f'{h["label"]} ({h["term"]})' for h in c["hpo_terms"]), c.get("free_text") or "",
                        c["stage1"]["cluster_name"], c["stage1"]["distance"], c["stage1"]["outlier"],
                        c["stage2"]["reliable"], c["nearest_known"] or "",
                        " | ".join(f'{k["disease"]}: {", ".join(k["failed_checks"])}' for k in c["candidates"]),
                        "; ".join(s["case_id"] for s in c["similar_cases"])])
        return buf.getvalue()


def case_from_result(r: dict, x: np.ndarray, meanings: list[str], context: dict | None = None) -> dict:
    """Pick the fields worth keeping from an Engine.identify() result."""
    s1, rec, s3 = r["stage1_neural"], r["stage2_fused_record"], r["stage3_symbolic"]
    ctx = context or {}
    cands = [{k: c[k] for k in ("disease", "score", "core_coverage", "failed_checks", "excluding_present",
                                "missing_core_symptoms")} for c in s3["candidates"]]
    return {"symptoms": [meanings[i] for i in np.flatnonzero(x)],
            "submitted": ctx.get("submitted"), "free_text": ctx.get("text"),
            "filled_defaults": ctx.get("filled_defaults", []),
            "hpo_terms": s3["grounding"],
            "stage1": {k: s1[k] for k in ("cluster_id", "cluster_name", "distance", "cluster_radius", "outlier")},
            "stage2": {"reliable": rec["reliable"], "unreliable_reasons": rec["unreliable_reasons"],
                       "rule": rec["rule"]["text"], "fidelity": rec["fidelity"]},
            "candidates": cands, "nearest_known": cands[0]["disease"] if cands else None,
            "reason": s3["resolution"]["reason"],
            "model": ctx.get("model", {})}
