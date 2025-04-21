import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def minimal_payload():
    return {
        "topic": "diabetes",
        "labeled_data": {
            "The patient has type 2 diabetes.": True,
            "He is feeling dizzy lately.": False
        }
    }

@pytest.fixture
def sufficient_payload():
    return {
        "topic": "asthma",
        "labeled_data": {
            f"Sentence {i}": bool(i % 2) for i in range(10)
        }
    }

def test_zero_shot_response(minimal_payload):
    response = client.post("/recommend", json=minimal_payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

def test_setfit_response(sufficient_payload):
    response = client.post("/recommend", json=sufficient_payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

def test_empty_payload():
    response = client.post("/recommend", json={"topic": "test", "labeled_data": {}})
    assert response.status_code == 400
    assert response.json()["detail"] == "Labeled data cannot be empty."

def test_invalid_input():
    response = client.post("/recommend", json={"foo": "bar"})
    assert response.status_code == 422
