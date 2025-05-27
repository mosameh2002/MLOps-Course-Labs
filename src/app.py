"""
FastAPI application for bank consumer churn prediction.
"""

import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import time
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="API for predicting bank customer churn",
    version="1.0.0"
)

# Load the model
MODEL_PATH = "C:/Users/mo/Desktop/MLops/MLOps-Course-Labs/model.pkl" 

try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


class ChurnPredictionRequest(BaseModel):
    """Request model for churn prediction."""
    CreditScore: int = Field(..., example=600)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., example=35)
    Tenure: int = Field(..., example=5)
    Balance: float = Field(..., example=50000.0)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=75000.0)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests."""
    request_id = str(time.time())
    logger.info(f"Request {request_id} started: {request.method} {request.url}")
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"Request {request_id} completed in {process_time:.4f}s with status {response.status_code}")
    return response


@app.get("/")
async def home():
    """Home endpoint."""
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Bank Customer Churn Prediction API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    if model is None:
        logger.error("Health check failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=Dict[str, Any])
async def predict(request: ChurnPredictionRequest):
    """
    Predict customer churn based on input features.
    
    Returns:
        Dict with prediction results
    """
    logger.info("Predict endpoint accessed")
    
    if model is None:
        logger.error("Prediction failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        logger.debug(f"Input data: {input_data}")
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        result = {
            "churn_prediction": int(prediction[0]),
            "churn_probability": float(prediction_proba[0][1]),
            "input_features": request.dict()
        }
        
        logger.info(f"Prediction successful: {result['churn_prediction']}")
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
