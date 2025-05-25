"""
Tests for the FastAPI application.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add the src directory to the path so we can import the app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from app import app

client = TestClient(app)

def test_home_endpoint():
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the Bank Customer Churn Prediction API" in response.json()["message"]

def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200 or response.status_code == 503
    # If model is loaded, status should be healthy
    if response.status_code == 200:
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

def test_predict_endpoint():
    """Test the predict endpoint."""
    # Skip this test if the model is not loaded
    health_response = client.get("/health")
    if health_response.status_code == 503:
        pytest.skip("Model not loaded, skipping prediction test")
    
    # Test data
    test_data = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "input_features" in result
    
    # Validate prediction type
    assert isinstance(result["churn_prediction"], int)
    assert isinstance(result["churn_probability"], float)
    assert 0 <= result["churn_probability"] <= 1