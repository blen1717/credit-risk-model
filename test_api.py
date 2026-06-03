import sys
sys.path.append('src/api')  # so we can import main from src/api

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

def test_predict_valid():
    # Mock the model's predict_proba method to avoid loading real model
    with patch('main.model') as mock_model:
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        response = client.post("/predict", json={"features": [0.0] * 10})
        assert response.status_code == 200
        assert "risk_probability" in response.json()

def test_predict_invalid():
    response = client.post("/predict", json={"features": [1, 2, 3]})
    assert response.status_code == 400
