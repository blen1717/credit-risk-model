import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict_valid():
    response = client.post("/predict", json={"features": [0.0] * 10})
    assert response.status_code == 200
    assert "risk_probability" in response.json()

def test_predict_invalid():
    response = client.post("/predict", json={"features": [1, 2, 3]})
    assert response.status_code == 400
