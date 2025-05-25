# Bank Customer Churn Prediction API

This project provides a FastAPI application for predicting bank customer churn.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train and save the model:

```bash
python src/train_and_save_model.py
```

3. Run the API:

```bash
python src/app.py
```

Or using uvicorn directly:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- **Home**: `/` - Welcome message
- **Health Check**: `/health` - Check if the API and model are working
- **Predict**: `/predict` - Make churn predictions

## Example Usage

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

## Running Tests

```bash
pytest tests/
```