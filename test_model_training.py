"""
Tests for the model training pipeline.
"""

import os
import pandas as pd
import pytest
import numpy as np
from src.train import preprocess, train
from src.train_and_save_model import train_and_save_model

@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'CustomerId': range(1, n_samples + 1),
        'Surname': ['Smith'] * n_samples,
        'CreditScore': np.random.randint(300, 900, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 95, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 5, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples),
        'IsActiveMember': np.random.choice([0, 1], n_samples),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
        'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def test_preprocessing_output_shapes(sample_data):
    """Test that preprocessing returns correctly shaped outputs."""
    # Run preprocessing
    col_transf, X_train, X_test, y_train, y_test = preprocess(sample_data)
    
    # Check that train/test split was performed
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]
    
    # Check that the total number of samples is preserved
    assert X_train.shape[0] + X_test.shape[0] == sample_data.shape[0]
    
    # Check that feature engineering created the expected number of features
    # This will depend on your preprocessing steps, adjust as needed
    assert X_train.shape[1] > 0, "No features were created during preprocessing"

def test_model_prediction_format(sample_data):
    """Test that the trained model produces predictions in the expected format."""
    # Preprocess data
    col_transf, X_train, X_test, y_train, y_test = preprocess(sample_data)
    
    # Train model
    model = train(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Check prediction shapes
    assert y_pred.shape == y_test.shape, "Predictions shape doesn't match target shape"
    assert y_pred_proba.shape[0] == y_test.shape[0], "Probability predictions count doesn't match target count"
    assert y_pred_proba.shape[1] == 2, "Binary classification should have 2 probability columns"
    
    # Check prediction values
    assert np.all(np.isin(y_pred, [0, 1])), "Predictions should be binary (0 or 1)"
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilities should be between 0 and 1"