"""Build the phenopacket-store subset used by the pipeline.

    python -m scripts.build_phenopackets <path-to-unzipped-release-folder>

* Diseases with >= 50 cases; 3 held out as "new" (seed 42).
* Each case's present HPO terms are expanded with their HPO ancestors, stopping
  below very general terms (the top two levels under Phenotypic abnormality),
  so related phenotypes share features.
* Features: expanded terms seen in >= 10 training cases and <= 60 % of them.
* Known diseases split 60/20/20 into train/validation/test per disease.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter

import numpy as np

from app.config import DATA_DIR, SEED
from app.stage3_symbolic.hpo import PHENOTYPIC_ABNORMALITY, get_hpo

MIN_CASES, N_HELD, MIN_FEATURE_CASES, MAX_FEATURE_SHARE = 50, 3, 10, 0.6


def main(root: str):
    hpo = get_hpo()
    rng = np.random.default_rng(SEED)
    pheno = hpo.descendants(PHENOTYPIC_ABNORMALITY)
    top = {c for c in hpo.children.get(PHENOTYPIC_ABNORMALITY, [])}
    too_general = {PHENOTYPIC_ABNORMALITY} | top | {g for c in top for g in hpo.children.get(c, [])}
    cases = []
    release = root.rstrip("/").split("/")[-1]
    for f in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
        j = json.load(open(f))
        pos = {p["type"]["id"] for p in j.get("phenotypicFeatures", []) if not p.get("excluded")}
        pos = {t for t in pos if t in pheno}
        if not pos or len(j.get("diseases", [])) != 1:
            continue
        exp = set(pos)
        for t in pos:
            exp |= (hpo.ancestors(t) & pheno) - too_general
        cases.append({"id": j["id"], "disease": j["diseases"][0]["term"]["label"],
                      "disease_id": j["diseases"][0]["term"]["id"], "observed": sorted(pos), "terms": exp})
    dc = Counter(c["disease"] for c in cases)
    keep = sorted(d for d, n in dc.items() if n >= MIN_CASES)
    held = sorted(rng.choice(keep, N_HELD, replace=False).tolist())
    known = [d for d in keep if d not in held]
    split = {}
    for d in keep:
        idx = [i for i, c in enumerate(cases) if c["disease"] == d]
        rng.shuffle(idx)
        if d in held:
            for i in idx:
                split[i] = "test"
        else:
            a, b = int(len(idx) * 0.6), int(len(idx) * 0.8)
            for k, i in enumerate(idx):
                split[i] = "train" if k < a else ("val" if k < b else "test")
    tr = [i for i, s in split.items() if s == "train"]
    tc = Counter(t for i in tr for t in cases[i]["terms"])
    feats = sorted(t for t, n in tc.items() if n >= MIN_FEATURE_CASES and n / len(tr) <= MAX_FEATURE_SHARE)
    fi = {t: k for k, t in enumerate(feats)}
    ids = sorted(split)
    X = np.zeros((len(ids), len(feats)), np.uint8)
    for r, i in enumerate(ids):
        for t in cases[i]["terms"]:
            if t in fi:
                X[r, fi[t]] = 1
    meta = {"features": feats, "meanings": [hpo.name(t) for t in feats],
            "cases": [{"id": cases[i]["id"], "disease": cases[i]["disease"], "disease_id": cases[i]["disease_id"],
                       "split": split[i], "observed": cases[i]["observed"]} for i in ids],
            "known_diseases": known, "heldout_new_diseases": held, "seed": SEED, "release": release,
            "hpo_release": hpo.data_version,
            "criteria": f">= {MIN_CASES} cases per disease; ancestors added below the top two HPO levels; "
                        f"features in >= {MIN_FEATURE_CASES} training cases and <= {MAX_FEATURE_SHARE:.0%} of them",
            "source": f"phenopacket-store {release} (Danis et al., HGG Advances 2025, 6(1):100371; "
                      f"Zenodo 10.5281/zenodo.13168726), real published case reports"}
    out = DATA_DIR / "phenopackets"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "X.npz", X=X)
    (out / "meta.json").write_text(json.dumps(meta))
    sp = Counter(split.values())
    print(f"cases kept {len(ids)} | known {len(known)} held {held} | split {dict(sp)} | features {len(feats)}")
    print("terms per case after expansion: median", int(np.median(X.sum(1))), "| empty rows:", int((X.sum(1) == 0).sum()))


if __name__ == "__main__":
    main(sys.argv[1])
