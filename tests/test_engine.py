import numpy as np

from dataset_loader import load_dataset
from engine import INCONCLUSIVE, KNOWN, NOVEL, EngineConfig, NSCAEngine, evaluate_holdout
from esoinn import ESOINN
from extractor import extract_symptoms
from fixtures import CORES, write_synthetic_csv


def build(tmp_path):
    ds = load_dataset(write_synthetic_csv(tmp_path / "s.csv"), tmp_path / "processed")
    # The tiny synthetic fixture has few rows per prototype, so associate at cluster level here.
    return ds, NSCAEngine(ds, EngineConfig(association_level="cluster", min_support=5))


def test_esoinn_finds_separate_clusters():
    rng = np.random.default_rng(0)
    protos = [(rng.random(15) < 0.35).astype(float) for _ in range(3)]
    X = []
    for p in protos:
        for _ in range(60):
            v = p.copy()
            flip = rng.random(15) < 0.08
            v[flip] = 1 - v[flip]
            X.append(v)
    m = ESOINN().fit(np.array(X), epochs=5)
    assert m.n_nodes >= 3 and m.n_clusters >= 2


def test_engine_trains_on_loaded_dataset(tmp_path):
    ds, eng = build(tmp_path)
    assert eng.model.n_nodes > 0 and eng.dataset is ds
    assert all(k in ds.symptom_index for k in ("sym_00", "sym_14"))


def test_known_pattern_single_disease(tmp_path):
    ds, eng = build(tmp_path)
    r = eng.identify([f"sym_{k:02d}" for k in CORES["disease_b"]])
    assert r.outcome == KNOWN and r.disease == "disease_b"
    assert r.cluster_id is not None and r.cluster_support >= 3


def test_insufficient_evidence_is_inconclusive(tmp_path):
    _, eng = build(tmp_path)
    r = eng.identify(["sym_00"])
    assert r.outcome == INCONCLUSIVE and r.disease is None
    assert eng.identify(["not_a_dataset_symptom", "another"]).outcome == INCONCLUSIVE


def test_unfamiliar_combination_flagged_not_diagnosed(tmp_path):
    _, eng = build(tmp_path)
    mix = [f"sym_{k:02d}" for k in (0, 1, 5, 6, 10, 11, 15, 16, 17)]
    r = eng.identify(mix)
    assert r.outcome in (NOVEL, INCONCLUSIVE) and r.disease is None
    if r.outcome == NOVEL:
        assert r.human_review_required and "not evidence of a new disease" in r.explanation


def test_never_returns_more_than_one_disease(tmp_path):
    _, eng = build(tmp_path)
    r = eng.identify(["sym_00", "sym_01", "sym_05"])
    assert r.disease is None or isinstance(r.disease, str)
    assert not any(isinstance(v, list) and v and r.disease in v for v in vars(r).values() if r.disease)


def test_holdout_evaluation_runs(tmp_path):
    ds, _ = build(tmp_path)
    ev = evaluate_holdout(ds, EngineConfig(association_level="cluster", min_support=5))
    part = evaluate_holdout(ds, EngineConfig(association_level="cluster", min_support=5), partial_k=3)
    assert part["performed"] and part["n_test"] == ev["n_test"]
    assert ev["performed"] and ev["n_test"] > 0
    assert ev["known_pattern"] + ev["potentially_novel_pattern"] + ev["inconclusive"] == ev["n_test"]


def test_extractor_matches_and_negates():
    feats = ["high_fever", "fever", "skin_rash", "cough"]
    matched, negated = extract_symptoms("I have a high fever and a bad cough, but no skin rash.", feats)
    assert matched == ["high_fever", "cough"] and negated == ["skin_rash"]
    assert extract_symptoms("feeling fine", feats) == ([], [])


def test_closest_association_only_on_non_known(tmp_path):
    _, eng = build(tmp_path)
    known = eng.identify([f"sym_{k:02d}" for k in CORES["disease_b"]])
    assert known.outcome == KNOWN and known.closest is None
    weak = eng.identify(["sym_00", "sym_05"])
    assert weak.outcome != KNOWN
    assert weak.closest is None or (isinstance(weak.closest["disease"], str) and 0 < weak.closest["similarity"] <= 1)
    assert eng.identify(["not_a_symptom"]).closest is None
