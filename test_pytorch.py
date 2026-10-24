
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_pytorch_prediction():
    response = client.post(
        "/predict-pytorch",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["predicted_species"] == "setosa"
    assert 0 <= data["confidence"] <= 1


def test_pytorch_rejects_invalid_input():
    response = client.post(
        "/predict-pytorch",
        json={
            "sepal_length": -1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
