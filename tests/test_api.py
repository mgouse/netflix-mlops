"""
Unit tests for Netflix API
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.inference.api import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Netflix Recommendation API"

def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_loaded" in response.json()

def test_predict_invalid_user():
    """Test prediction with invalid user"""
    response = client.post(
        "/predict",
        json={"user_id": "invalid_user", "n_recommendations": 5}
    )
    assert response.status_code == 404

def test_predict_valid_user():
    """Test prediction with valid user"""
    # This test will only work if model is loaded
    # In CI, we might mock this
    response = client.post(
        "/predict",
        json={"user_id": "user_0", "n_recommendations": 5}
    )
    # If model is loaded, should be 200
    # If not loaded in test env, might be 503
    assert response.status_code in [200, 503]

def test_production_stats():
    """Test production stats endpoint"""
    response = client.get("/production-stats")
    assert response.status_code == 200
    assert "total_predictions" in response.json()
    assert "error_rate" in response.json()