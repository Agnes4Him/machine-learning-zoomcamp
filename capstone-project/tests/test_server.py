import pytest
from fastapi.testclient import TestClient
from server.predict import app, predict_single, dv, rf, HouseholdFeatures

client = TestClient(app)

# Health Check Endpoint
def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "PONG"}

# Predict Endpoint (valid input)
'''def test_predict_valid():
    payload = {
        "appliance_type": "Oven",
        "season": "Winter",
        "outdoor_temperature": 22.5,
        "household_size": 4,
        "hour": 14,
        "day_of_week": 2,
        "day": 15,
        "month": 12,
        "is_weekend": 0
    }

    response = client.post("/predict", json=payload)
    
    # The endpoint should return 200
    assert response.status_code == 200
    json_data = response.json()
    assert "energy_consumption" in json_data
    assert isinstance(json_data["energy_consumption"], float)'''

# Predict Endpoint (invalid input)
'''def test_predict_invalid():
    # Missing required field 'hour'
    payload = {
        "appliance_type": "Oven",
        "season": "Winter",
        "outdoor_temperature": 22.5,
        "household_size": 4,
        "day_of_week": 2,
        "day": 15,
        "month": 12,
        "is_weekend": 0
    }

    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422
    assert "Invalid input" in response.json()["message"]'''

# Direct test of prediction logic
def test_predict_single_function(monkeypatch):
    # Patch dv and rf if not loaded
    if dv is None or rf is None:
        class DummyDV:
            def transform(self, x):
                return [[1]*9]  # fake vector
        class DummyRF:
            def predict(self, X):
                return [42.42]
        '''monkeypatch.setattr("app.main.dv", DummyDV())
        monkeypatch.setattr("app.main.rf", DummyRF())'''
        monkeypatch.setattr("dv", DummyDV())
        monkeypatch.setattr("rf", DummyRF())

    payload = {
        "appliance_type": "Oven",
        "season": "Winter",
        "outdoor_temperature": 22.5,
        "household_size": 4,
        "hour": 14,
        "day_of_week": 2,
        "day": 15,
        "month": 12,
        "is_weekend": 0
    }

    result = predict_single(payload)
    assert isinstance(result, float)