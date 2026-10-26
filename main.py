
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import numpy as np

# Load model when the server starts
ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["predictor"] = joblib.load("model.pkl")
    print("Model loaded successfully")

    yield

    ml_model.clear()


app = FastAPI(title="Diabetes Risk API", lifespan=lifespan)


# What the caller sends
class PatientFeatures(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# What we send back
class PredictionResult(BaseModel):
    prediction: int
    result_label: str
    confidence: float


# Health check
@app.get("/")
def health():
    return {
        "status": "running",
        "model_loaded": "predictor" in ml_model
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResult)
def predict(features: PatientFeatures):
    model = ml_model.get("predictor")

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    # Build input array in the same order as training
    X = np.array([[
        features.Pregnancies,
        features.Glucose,
        features.BloodPressure,
        features.SkinThickness,
        features.Insulin,
        features.BMI,
        features.DiabetesPedigreeFunction,
        features.Age
    ]])

    # Get prediction
    pred = int(model.predict(X)[0])

    # Get confidence of the predicted class
    probabilities = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(pred)
    conf = round(float(probabilities[class_index]), 3)

    return PredictionResult(
        prediction=pred,
        result_label="Diabetic" if pred == 1 else "Not Diabetic",
        confidence=conf
    )
