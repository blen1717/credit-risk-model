from pydantic import BaseModel

class PredictionRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float
    hour: float
    day_of_week: float
    month: float
    year: float

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_label: int
