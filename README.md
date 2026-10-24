# Iris ML Prediction API

A machine learning project that predicts Iris flower species using scikit-learn and PyTorch, exposed through a REST API built with FastAPI.

## Features

- Iris flower classification using scikit-learn and PyTorch
- Data preprocessing with `StandardScaler`
- Prediction confidence scores
- REST endpoints with FastAPI
- Input validation using Pydantic
- Health-check endpoint
- Automated tests using pytest and FastAPI's `TestClient`
- Saved model artifacts for inference

## Tech Stack

- Python
- PyTorch
- scikit-learn
- FastAPI
- Pandas
- NumPy
- Joblib
- Pytest

## Project Structure

```text
iris-ml-api/
├── main.py
├── train.py
├── train_pytorch.py
├── test_main.py
├── test_pytorch.py
├── iris_model.joblib
├── iris_scaler.joblib
├── iris_pytorch_model.pth
└── README.md
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install fastapi uvicorn scikit-learn pandas joblib pytest httpx torch
```

## Train the Models

Train the scikit-learn model:

```powershell
python train.py
```

Train the PyTorch model:

```powershell
python train_pytorch.py
```

Run both training scripts before starting the API so the required model files and scaler are available.

## Start the API

```powershell
python -m uvicorn main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check that the API is running |
| GET | `/health` | Check model and scaler loading |
| POST | `/predict` | Predict using scikit-learn |
| POST | `/predict-pytorch` | Predict using PyTorch |

### Example Request

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Send the JSON to either prediction endpoint using the interactive Swagger documentation.

## Run Tests

```powershell
python -m pytest -v
```

The project currently has six automated tests covering API responses, prediction, invalid input, health checks, and PyTorch inference.

## What I Learned

- Splitting data into training and test sets
- Scaling features without fitting on test data
- Training and evaluating a neural network
- Saving and loading trained models
- Building REST APIs for ML inference
- Validating input data
- Writing automated API tests
- Logging prediction activity

## Future Improvements

- Add model comparison and additional evaluation metrics
- Add CI testing with GitHub Actions
- Containerize the API with Docker
- Deploy the API to a cloud platform
