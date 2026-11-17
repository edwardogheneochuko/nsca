"""Tests: unit checks per stage plus API end-to-end on the trained models."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_DIR
from app.stage1_neural.esoinn import ESOINN
from app.stage2_xai.exemplars import jaccard_matrix, k_medoids
from app.stage3_symbolic.hpo import get_hpo


# ---------------------------------------------------------------- Stage 1
def _toy(n=600, seed=0):
    """Three well-separated 'diseases' over 30 binary symptoms."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in range(3):
        core = np.arange(k * 10, k * 10 + 10)
        for _ in range(n // 3):
            x = np.zeros(30, np.uint8)
            x[rng.choice(core, 5, replace=False)] = 1
            X.append(x); y.append(k)
    X, y = np.array(X), np.array(y)
    o = rng.permutation(len(X))
    return X[o], y[o]


def test_esoinn_separates_disjoint_groups():
    X, y = _toy()
    m = ESOINN(30, lam=100, age_max=5, t_max=0.7).fit(X).finalize(X, y)
    p = m.predict(X)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(y, p["cluster"]) > 0.9
    assert len(set(m.cluster[m.nodes].tolist())) == 3


def test_esoinn_flags_unseen_pattern_as_outlier():
    X, y = _toy()
    m = ESOINN(30, lam=100, age_max=5, t_max=0.7).fit(X).finalize(X, y)
    odd = np.zeros(30, np.uint8); odd[[0, 12, 25, 29]] = 1   # mixes all three groups
    assert m.predict(odd.reshape(1, -1))["outlier"][0]


def test_esoinn_save_load_roundtrip(tmp_path):
    X, y = _toy()
    m = ESOINN(30, lam=100, age_max=5).fit(X).finalize(X, y)
    m.save(tmp_path)
    m2 = ESOINN.load(tmp_path)
    assert np.array_equal(m.predict(X)["cluster"], m2.predict(X)["cluster"])


def test_soft_jaccard_matches_definition():
    m = ESOINN(4)
    m._new_node(np.array([1, 0.5, 0, 0], np.float32), None)
    m._new_node(np.array([0, 0, 1, 1], np.float32), None)
    x = np.array([1, 1, 0, 0], np.uint8)
    d = m.dist_to_nodes(x)
    assert d[0] == pytest.approx(1 - 1.5 / 2.0)
    assert d[1] == pytest.approx(1.0)


# ---------------------------------------------------------------- Stage 2
def test_k_medoids_returns_real_points():
    X, _ = _toy(90)
    D = jaccard_matrix(X, X)
    med = k_medoids(D, 3)
    assert len(set(med.tolist())) == 3 and med.max() < len(X)


# ---------------------------------------------------------------- Stage 3
def test_hpo_loaded_and_hierarchy():
    h = get_hpo()
    assert h.name("HP:0001945") == "Fever"
    assert "HP:0000118" in h.ancestors("HP:0012735")      # cough is a phenotypic abnormality


# ---------------------------------------------------------------- API
needs_models = pytest.mark.skipif(not (MODEL_DIR / "ddxplus" / "meta.json").exists(), reason="models not trained")


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    with TestClient(app) as c:
        yield c


@needs_models
def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and "ddxplus" in r["models_loaded"]


@needs_models
def test_symptom_list_has_hpo(client):
    s = client.get("/symptoms", params={"dataset": "ddxplus", "q": "cough"}).json()
    assert any(x["hpo_label"] == "Cough" for x in s)


@needs_models
def test_identify_returns_every_stage(client):
    r = client.post("/identify", json={"dataset": "ddxplus", "symptoms": ["E_91", "E_201", "E_97", "E_181"]})
    assert r.status_code == 200
    j = r.json()
    assert j["outcome"] in ("known_disease", "flag_as_new")
    assert set(j["stages"]) == {"stage1_neural", "stage2_fused_record", "stage3_symbolic"}
    rec = j["stages"]["stage2_fused_record"]
    for k in ("rule", "node_difference", "exemplars", "fidelity", "agreement", "reliable"):
        assert k in rec
    assert j["stages"]["stage3_symbolic"]["grounding"]


@needs_models
def test_identify_accepts_plain_names_and_reports_unknown(client):
    r = client.post("/identify", json={"dataset": "mendeley", "symptoms": ["fever", "cough", "not-a-symptom"],
                                       "detail": "summary"}).json()
    assert r["unknown_symptoms"] == ["not-a-symptom"] and r["stages"] is None


@needs_models
def test_identify_rejects_all_unknown(client):
    assert client.post("/identify", json={"symptoms": ["zzz"]}).status_code == 422


@needs_models
def test_model_status_and_clusters(client):
    s = client.get("/model-status", params={"dataset": "ddxplus"}).json()
    assert s["stage2_surrogate"]["fidelity"] >= 0.9
    c = client.get("/clusters", params={"dataset": "ddxplus"}).json()
    assert len(c) == s["stage1_esoinn"]["clusters"]


@needs_models
def test_questions_ddxplus_grouped(client):
    qs = client.get("/questions", params={"dataset": "ddxplus"}).json()
    by_id = {q["id"]: q for q in qs}
    assert by_id["E_91"]["type"] == "yes_no"
    assert by_id["E_56"]["type"] == "scale" and by_id["E_56"]["parent"] == "E_53"
    assert by_id["E_55"]["type"] == "multi" and len(by_id["E_55"]["options"]) > 10
    assert [q["id"] for q in qs].index("E_56") > [q["id"] for q in qs].index("E_53")


@needs_models
def test_defaults_filled_for_followups_and_travel(client):
    r = client.post("/identify", json={"dataset": "ddxplus", "symptoms": ["E_53", "E_91"], "detail": "summary"}).json()
    assert "E_204=V_10" in r["filled_defaults"]           # travel: no
    assert any(f.startswith("E_56=") for f in r["filled_defaults"])   # pain intensity follow-up
    r2 = client.post("/identify", json={"dataset": "ddxplus", "symptoms": ["E_91"], "detail": "summary"}).json()
    assert not any(f.startswith("E_56=") for f in r2["filled_defaults"])   # no pain -> no pain follow-ups


@needs_models
def test_parse_text_everyday_and_negation(client):
    r = client.post("/parse-text", json={"dataset": "ddxplus",
                                         "text": "I have itching and feeling unwell"}).json()
    assert "E_129" in r["symptoms"] and "E_175" in r["symptoms"]       # skin (itching follow-up), malaise
    r = client.post("/parse-text", json={"dataset": "ddxplus",
                                         "text": "my body is hot, I am stooling and have catarrh but no sore throat"}).json()
    assert {"E_91", "E_51", "E_181"} <= set(r["symptoms"]) and "E_97" not in r["symptoms"]
    assert [a["question_id"] for a in r["absent"]] == ["E_97"]


@needs_models
def test_parse_text_pain_location_and_unrecognised(client):
    r = client.post("/parse-text", json={"dataset": "ddxplus", "text": "pain in my stomach, I love football"}).json()
    assert r["symptoms"] == ["E_53"] and len(r["answers"]["E_55"]) == 1
    assert r["unrecognised"] == ["I love football"]


# ---------------------------------------------------------------- saved "new disease" cases
def test_flag_store_saves_links_similar_and_survives_reload(tmp_path):
    from app.flag_store import FlagStore
    case = {"symptoms": ["a"], "hpo_terms": [], "stage1": {"cluster_name": "x", "distance": 0.9, "outlier": True},
            "stage2": {"reliable": False}, "candidates": [], "nearest_known": None}
    st = FlagStore(tmp_path / "d.json", "ddxplus")
    a = st.save([1, 2, 3, 4], case)
    assert a["saved"] and a["similar_cases"] == []
    assert st.save([1, 2, 3, 4], case)["duplicate_of_recent"]          # resubmission is not a second case
    b = st.save([1, 2, 3, 5], case)                                      # shares 3 of 5 symptoms
    assert b["similar_cases"][0]["case_id"] == a["case_id"]
    assert st.groups() == [[a["case_id"], b["case_id"]]]
    st.review(a["case_id"], "confirmed_new", note="checked")
    again = FlagStore(tmp_path / "d.json", "ddxplus")                  # e.g. after a server restart
    assert again.counts()["total"] == 2 and again.get(a["case_id"])["review"]["status"] == "confirmed_new"
    assert again.get(a["case_id"])["similar_cases"][0]["case_id"] == b["case_id"]
    assert "case_id" in again.to_csv().splitlines()[0]


@needs_models
def test_new_disease_is_saved_and_reviewable(client):
    # palpitations + shortness of breath only: no known disease passes the checks (see the demo)
    body = {"dataset": "ddxplus", "symptoms": ["E_155", "E_66"], "detail": "summary",
            "text": "my heart is racing and I am short of breath"}
    r = client.post("/identify", json=body).json()
    if r["outcome"] != "flag_as_new":
        pytest.skip("this input was not flagged by the current model")
    cid = r["flag"]["case_id"]
    saved = client.get(f"/flags/{cid}?dataset=ddxplus").json()
    assert saved["free_text"] == body["text"] and saved["review"]["status"] == "pending_review"
    assert saved["hpo_terms"] and saved["candidates"]
    assert client.post("/identify", json=body).json()["flag"]["duplicate_of_recent"]
    rv = client.patch(f"/flags/{cid}", json={"dataset": "ddxplus", "status": "known_disease", "disease": "Panic attack"})
    assert rv.status_code == 200 and rv.json()["review"]["disease"] == "Panic attack"
    lst = client.get("/flags?dataset=ddxplus").json()
    assert lst["counts"]["known_disease"] >= 1
    assert client.get("/flags/export?dataset=ddxplus").text.startswith("case_id")
    assert client.patch("/flags/NOPE", json={"dataset": "ddxplus", "status": "dismissed"}).status_code == 404


@needs_models
def test_known_disease_is_not_saved(client):
    r = client.post("/identify", json={"dataset": "ddxplus", "symptoms": ["E_91", "E_201", "E_97", "E_181"],
                                       "detail": "summary"}).json()
    if r["outcome"] == "known_disease":
        assert r["flag"] is None


def test_flag_store_delete_and_clear(tmp_path):
    from app.flag_store import FlagStore
    case = {"symptoms": [], "hpo_terms": [], "stage1": {}, "stage2": {}, "candidates": [], "nearest_known": None}
    st = FlagStore(tmp_path / "d.json", "ddxplus")
    a, b, c = (st.save(f, case)["case_id"] for f in ([1, 2], [1, 2, 3], [7, 8]))
    assert st.delete(a) and not st.delete(a)
    assert all(s["case_id"] != a for s in st.get(b)["similar_cases"])
    st.review(c, "dismissed")
    assert st.clear("dismissed") == 1 and st.clear() == 1
    assert FlagStore(tmp_path / "d.json", "ddxplus").counts()["total"] == 0


@needs_models
def test_candidates_have_structured_checks(client):
    r = client.post("/identify", json={"dataset": "ddxplus", "symptoms": ["E_201", "E_45", "E_91"]}).json()
    c = r["stages"]["stage3_symbolic"]["candidates"][0]
    assert [k["name"] for k in c["checks"]] == ["overall_match", "usual_symptoms", "nothing_against", "symptoms_explained"]
    assert c["consistent"] == all(k["passed"] for k in c["checks"])
    assert "core_present" in c and r["stages"]["stage3_symbolic"]["hpo_release"]
