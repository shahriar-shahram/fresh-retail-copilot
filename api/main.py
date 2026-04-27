from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


MODEL_DIR = Path("models")
REGISTRY_PATH = MODEL_DIR / "forecast_model_registry.json"
METADATA_PATH = MODEL_DIR / "forecast_model_metadata.json"

feature_cols = joblib.load(MODEL_DIR / "forecast_features.pkl")


def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {}


def load_registry():
    metadata = load_metadata()

    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    if not registry:
        registry = {
            "default": {
                "display_name": metadata.get("selected_model", "Default model"),
                "model_class": metadata.get("model_class", "Unknown"),
                "path": str(MODEL_DIR / "forecast_model.pkl"),
                "mae": None,
                "rmse": None,
                "smape": None,
                "rank_by_mae": 1,
            }
        }

    default_key = metadata.get("selected_model_key")
    if default_key is None or default_key not in registry:
        default_key = sorted(registry.items(), key=lambda item: item[1].get("rank_by_mae", 999))[0][0]

    return registry, default_key, metadata


registry, default_model_key, metadata = load_registry()
model_cache = {}


def get_model(model_key: str | None):
    key = model_key or default_model_key

    if key not in registry:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown model_name '{key}'",
                "available_models": list(registry.keys()),
                "default_model": default_model_key,
            },
        )

    if key not in model_cache:
        model_cache[key] = joblib.load(registry[key]["path"])

    return key, model_cache[key]


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
    model_name: str | None = None


app = FastAPI(title="Fresh Retail Copilot API")


@app.get("/")
def root():
    return {
        "message": "Fresh Retail Copilot API",
        "default_model": default_model_key,
        "available_models": list(registry.keys()),
    }


@app.get("/models")
def list_models():
    return {
        "default_model": default_model_key,
        "models": registry,
        "metadata": metadata,
    }


@app.post("/predict")
def predict(req: ForecastRequest):
    payload = req.model_dump()
    requested_model = payload.pop("model_name", None)

    model_key, model = get_model(requested_model)

    X = pd.DataFrame([payload], columns=feature_cols)
    prediction = model.predict(X)[0]
    prediction = max(float(prediction), 0.0)

    model_info = registry[model_key]

    return {
        "predicted_demand": prediction,
        "model_name": model_key,
        "model_display_name": model_info.get("display_name", model_key),
        "model_class": model_info.get("model_class"),
        "mae": model_info.get("mae"),
        "rmse": model_info.get("rmse"),
        "smape": model_info.get("smape"),
        "rank_by_mae": model_info.get("rank_by_mae"),
    }

# -----------------------------
# Business horizon forecast
# -----------------------------

BUSINESS_REGISTRY_PATH = MODEL_DIR / "business_horizon_model_registry.json"
BUSINESS_METADATA_PATH = MODEL_DIR / "business_horizon_metadata.json"
BUSINESS_FEATURES_PATH = MODEL_DIR / "business_horizon_features.pkl"

business_model_cache = {}


def load_business_horizon_registry():
    if not BUSINESS_REGISTRY_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Business horizon model registry not found. Run scripts/train_business_horizon_models.py first.",
        )

    with open(BUSINESS_REGISTRY_PATH, "r") as f:
        registry = json.load(f)

    metadata = {}
    if BUSINESS_METADATA_PATH.exists():
        with open(BUSINESS_METADATA_PATH, "r") as f:
            metadata = json.load(f)

    features = joblib.load(BUSINESS_FEATURES_PATH)

    return registry, metadata, features


class BusinessHorizonForecastRequest(BaseModel):
    city_id: int
    store_id: int
    management_group_id: int
    first_category_id: int
    second_category_id: int
    third_category_id: int
    product_id: int

    lag_1: float
    lag_2: float
    lag_3: float
    lag_7: float
    lag_14: float
    lag_21: float
    lag_28: float

    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_28: float
    rolling_std_7: float
    rolling_std_14: float
    rolling_std_28: float
    rolling_min_7: float
    rolling_max_7: float
    rolling_min_14: float
    rolling_max_14: float
    rolling_min_28: float
    rolling_max_28: float

    stock_hour6_22_cnt: float
    in_stock_ratio: float
    stockout_hours: float
    avg_sales_when_available: float

    discount: float
    holiday_flag: int
    activity_flag: int

    precpt: float
    avg_temperature: float
    avg_humidity: float
    avg_wind_level: float

    day_of_week: int
    month: int
    day_of_month: int
    week_of_year: int
    is_weekend: int

    horizon: str = "1_week_ahead"


@app.get("/business-horizon-models")
def business_horizon_models():
    registry, metadata, features = load_business_horizon_registry()
    return {
        "registry": registry,
        "metadata": metadata,
        "features": features,
        "available_horizons": list(registry.keys()),
    }


@app.post("/predict-business-horizon")
def predict_business_horizon(req: BusinessHorizonForecastRequest):
    registry, metadata, features = load_business_horizon_registry()

    payload = req.model_dump()
    horizon = payload.pop("horizon")

    if horizon not in registry:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown horizon '{horizon}'",
                "available_horizons": list(registry.keys()),
            },
        )

    model_info = registry[horizon]
    model_path = model_info["model_path"]

    if horizon not in business_model_cache:
        business_model_cache[horizon] = joblib.load(model_path)

    model = business_model_cache[horizon]

    X = pd.DataFrame([payload], columns=features)
    prediction = model.predict(X)[0]
    prediction = max(float(prediction), 0.0)

    return {
        "horizon": horizon,
        "horizon_steps": model_info.get("horizon_steps"),
        "predicted_demand": prediction,
        "model_name": model_info.get("selected_model"),
        "model_class": model_info.get("model_class"),
        "mae": model_info.get("mae"),
        "rmse": model_info.get("rmse"),
        "smape": model_info.get("smape"),
        "target": model_info.get("target"),
    }
