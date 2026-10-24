
import joblib
import torch
import logging
import pandas as pd

from fastapi import FastAPI
from torch import nn
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


saved = joblib.load("iris_model.joblib")
scaler = joblib.load("iris_scaler.joblib")
model = saved["model"]
target_names = saved["target_names"]
feature_names = saved["feature_names"]

app = FastAPI(title="Iris ML Prediction API")


class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Iris ML API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "sklearn_model_loaded": model is not None,
        "pytorch_model_loaded": torch_model is not None,
        "scaler_loaded": scaler is not None
    }
    
@app.post("/predict")
def predict_species(flower: FlowerInput):
    row = pd.DataFrame(
        [[
            flower.sepal_length,
            flower.sepal_width,
            flower.petal_length,
            flower.petal_width
        ]],
        columns=feature_names
    )

    prediction = int(model.predict(row.to_numpy())[0])
    probabilities = model.predict_proba(row.to_numpy())[0]

    return {
        "predicted_species": target_names[prediction],
        "confidence": round(float(probabilities[prediction]), 4)
    }

# PyTorch model architecture — must match training
torch_model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

torch_model.load_state_dict(
    torch.load("iris_pytorch_model.pth", map_location="cpu", weights_only=True)
)
torch_model.eval()


@app.post("/predict-pytorch")
def predict_pytorch(flower: FlowerInput):
    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    # Use the same feature scaling as training
    scaled_features = scaler.transform(features)

    tensor_input = torch.tensor(
        scaled_features, dtype=torch.float32
    )

    logger.info("Received PyTorch prediction request") 

    with torch.no_grad():
        logits = torch_model(tensor_input)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

    logger.info("Prediction completed: %s", target_names[predicted_class])

    return {
        "model": "PyTorch",
        "predicted_species": target_names[predicted_class],
        "confidence": round(confidence, 4)
    }
