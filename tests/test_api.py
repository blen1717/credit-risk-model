import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Risk API is running"}

def test_predict():
    sample = {
        "recency": 100,
        "frequency": 5,
        "monetary": 5000,
        "hour": 14,
        "day_of_week": 3,
        "month": 6,
        "year": 2019
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert "risk_probability" in response.json()
    assert "risk_label" in response.json()
