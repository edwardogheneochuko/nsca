import pytest
from fastapi.testclient import TestClient

from api import create_app
from fixtures import write_synthetic_csv


@pytest.fixture
def client(tmp_path):
    csv = write_synthetic_csv(tmp_path / "s.csv")
    with TestClient(create_app(str(csv), str(tmp_path / "processed"))) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/health").json() == {"status": "ok", "dataset_loaded": True}
    s = client.get("/model/status").json()
    assert s["dataset_loaded"] and s["esoinn_trained"]
    assert s["n_diseases"] == 3 and s["n_symptom_features"] > 0 and s["n_training_rows"] > 0
    assert s["evaluation"]["performed"] is True


def test_extract_and_identify(client):
    r = client.post("/symptoms/extract", json={"description": "I have sym 00 and sym 01 but no sym 02"})
    assert r.json() == {"matched_symptoms": ["sym_00", "sym_01"], "negated_symptoms": ["sym_02"]}
    r = client.post("/identify", json={"description": "sym 05, sym 06, sym 07, sym 08", "severity": "severe"})
    body = r.json()
    assert r.status_code == 200 and body["outcome"] in ("known_pattern", "potentially_novel_pattern", "inconclusive")
    assert "urgent" in body["urgent_care_notice"].lower() and body["disclaimer"]
    assert "diseases" not in body and "ranking" not in body


def test_identify_with_no_recognised_symptoms_is_inconclusive(client):
    body = client.post("/identify", json={"description": "I feel strange"}).json()
    assert body["outcome"] == "inconclusive" and body["disease"] is None


def test_validation_error(client):
    assert client.post("/identify", json={"description": ""}).status_code == 422
    assert client.post("/identify", json={"description": "x", "severity": "extreme"}).status_code == 422


def test_503_when_dataset_missing(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing.csv"), str(tmp_path / "p"))) as c:
        assert c.get("/health").json()["dataset_loaded"] is False
        r = c.get("/model/status")
        assert r.status_code == 503 and "not found" in r.json()["detail"]
        assert c.post("/identify", json={"description": "cough"}).status_code == 503


def test_symptom_list_and_explicit_symptoms(client):
    names = client.get("/symptoms").json()["symptoms"]
    assert "sym_00" in names and names == sorted(names)
    body = client.post("/identify", json={"description": "picked", "symptoms": ["sym_05", "sym_06", "sym_07"]}).json()
    assert body["observed_symptoms"] == ["sym_05", "sym_06", "sym_07"]
