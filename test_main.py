
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_flower():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200

    data = response.json()
    assert data["predicted_species"] in [
        "setosa", "versicolor", "virginica"
    ]
    assert 0 <= data["confidence"] <= 1


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 422


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["sklearn_model_loaded"] is True
    assert data["pytorch_model_loaded"] is True
    assert data["scaler_loaded"] is True
