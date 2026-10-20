from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_extract_symptoms():
    response = client.post("/api/v1/symptoms/extract", json={"description": "I have fever and headache"})
    assert response.status_code == 200
    assert set(response.json()["symptoms"]) == {"fever", "headache"}

def test_inconclusive_for_unrecognized_text():
    response = client.post("/api/v1/identify", json={"description": "I feel strange"})
    assert response.status_code == 200
    assert response.json()["outcome"] == "inconclusive"

def test_identification_returns_one_outcome():
    response = client.post("/api/v1/identify", json={"description": "I have fever, cough, sore throat and body aches"})
    assert response.status_code == 200
    assert response.json()["outcome"] in {"known_pattern", "potentially_novel_pattern", "inconclusive"}
    assert "disease_candidates" not in response.json()
    assert response.json()["review_required"] is True

def test_urgent_symptoms_are_not_diagnosed():
    response = client.post("/api/v1/identify", json={"description": "I have chest pain and shortness of breath"})
    assert response.status_code == 200
    assert response.json()["outcome"] == "inconclusive"
    assert response.json()["disease"] is None
