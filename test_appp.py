from fastapi.testclient import TestClient
from app.appp import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from FastAPI + Docker!"}
