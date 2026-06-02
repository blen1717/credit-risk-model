import pandas as pd
import joblib
from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse

# Load model
model = joblib.load('best_model.pkl')

app = FastAPI(title="Credit Risk API")

@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])
    proba = model.predict_proba(input_df)[0, 1]
    label = int(proba >= 0.5)
    return PredictionResponse(risk_probability=proba, risk_label=label)
