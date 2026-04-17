from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Fresh Retail Copilot API")

model = joblib.load("models/forecast_model.pkl")
feature_cols = joblib.load("models/forecast_features.pkl")


class ForecastRequest(BaseModel):
    lag_1: float
    lag_7: float
    rolling_mean_7: float
    discount: float
    holiday_flag: int
    activity_flag: int
    precpt: float
    avg_temperature: float
    avg_humidity: float
    avg_wind_level: float
    day_of_week: int
    month: int
    avg_sales_when_available: float
    stockout_hours: int
    demand_std: float


@app.get("/")
def root():
    return {"message": "Fresh Retail Copilot API is running"}


@app.post("/predict")
def predict(req: ForecastRequest):
    X = pd.DataFrame([req.model_dump()], columns=feature_cols)
    prediction = model.predict(X)[0]
    prediction = max(float(prediction), 0.0)
    return {"predicted_demand": prediction}