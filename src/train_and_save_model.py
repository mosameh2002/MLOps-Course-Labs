"""
Script to train and save the churn prediction model for use with the API.
"""

import os
import logging
import mlflow
import mlflow.sklearn
from train import preprocess, train
import pandas as pd
import hyperdx  # Import HyperDX

# Configure HyperDX logging
HYPERDX_API_KEY = os.getenv("HYPERDX_API_KEY")
if HYPERDX_API_KEY:
    hyperdx_handler = hyperdx.HyperDXHandler(
        api_key=HYPERDX_API_KEY,
        service_name="churn-prediction-training",
        level=logging.INFO
    )
else:
    hyperdx_handler = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add HyperDX handler if available
if hyperdx_handler:
    logger.addHandler(hyperdx_handler)
    logger.info("HyperDX logging enabled")
else:
    logger.info("HyperDX logging not configured")

def train_and_save_model():
    """Train the model and save it for the API to use."""
    logger.info("Starting model training")
    
    # Create models directory if it doesn't exist
    os.makedirs("models/latest", exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading data")
        df = pd.read_csv("C:\\Users\\mo\\Desktop\\MLops\\MLOps-Course-Labs\\data\\Churn_Modelling.csv")
        
        # Log data statistics
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Churn rate: {df['Exited'].mean():.2f}")
        
        # Preprocess data
        logger.info("Preprocessing data")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        
        # Train model
        logger.info("Training model")
        model = train(X_train, y_train)
        
        # Save column transformer and model
        logger.info("Saving model and transformer")
        mlflow.sklearn.save_model(model, "models/latest")
        
        # Also save the column transformer
        mlflow.sklearn.save_model(col_transf, "models/latest/preprocessor")
        
        logger.info("Model training and saving completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in train_and_save_model: {str(e)}")
        return False

if __name__ == "__main__":
    train_and_save_model()
