import json
import re
from pathlib import Path
import importlib.util

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


TRAIN_PATH = Path("data/processed/model_train_rich.parquet")

MODEL_DIR = Path("models")
REGISTRY_DIR = MODEL_DIR / "business_horizon_registry"
RESULTS_DIR = Path("results/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = {
    "1_day_ahead": 1,
    "1_week_ahead": 7,
    "1_month_ahead": 30,
}

ID_COLS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

FEATURE_COLS = [
    # ID/category features
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",

    # Demand history
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",

    # Rolling demand
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_min_7",
    "rolling_max_7",
    "rolling_min_14",
    "rolling_max_14",
    "rolling_min_28",
    "rolling_max_28",

    # Availability
    "stock_hour6_22_cnt",
    "in_stock_ratio",
    "stockout_hours",
    "avg_sales_when_available",

    # Business context
    "discount",
    "holiday_flag",
    "activity_flag",

    # Weather
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",

    # Calendar
    "day_of_week",
    "month",
    "day_of_month",
    "week_of_year",
    "is_weekend",
]


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0

    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "dt"]).copy()
    g = df.groupby(["store_id", "product_id"], group_keys=False)

    for label, h in HORIZONS.items():
        df[f"target_{label}"] = g["sale_amount"].shift(-h)

    return df


def load_data():
    use_cols = ["dt", "sale_amount"] + FEATURE_COLS
    df = pd.read_parquet(TRAIN_PATH, columns=use_cols)
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    for col in ["sale_amount"] + FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["dt", "sale_amount"] + FEATURE_COLS)

    df = add_targets(df)

    return df


def candidate_models():
    models = {
        "Naive mean baseline": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=450,
            learning_rate=0.05,
            l2_regularization=0.05,
            max_leaf_nodes=31,
        ),
    }

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=550,
            learning_rate=0.035,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            objective="regression",
        )

    if importlib.util.find_spec("catboost") is not None:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=600,
            learning_rate=0.04,
            depth=8,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def evaluate_baselines(train_df, valid_df, target_col):
    y_valid = valid_df[target_col].to_numpy(dtype=float)

    baselines = {
        "Last observed value baseline": valid_df["lag_1"].to_numpy(dtype=float),
        "7-period average baseline": valid_df["rolling_mean_7"].to_numpy(dtype=float),
        "Seasonal lag-7 baseline": valid_df["lag_7"].to_numpy(dtype=float),
        "Seasonal lag-28 baseline": valid_df["lag_28"].to_numpy(dtype=float),
    }

    rows = []

    for name, pred in baselines.items():
        pred = np.maximum(pred, 0.0)
        rows.append(
            {
                "model": name,
                "model_key": safe_name(name),
                "model_class": "rule_based_baseline",
                "mae": float(mean_absolute_error(y_valid, pred)),
                "rmse": float(root_mean_squared_error(y_valid, pred)),
                "smape": smape(y_valid, pred),
                "is_trainable_model": False,
            }
        )

    return rows


def train_one_horizon(df, horizon_label, horizon_steps):
    target_col = f"target_{horizon_label}"

    work = df[["dt"] + FEATURE_COLS + [target_col]].dropna(subset=FEATURE_COLS + [target_col]).copy()

    # Horizon-aware time split.
    # Longer horizons have fewer usable target dates, so a fixed 60-day holdout
    # can accidentally leave zero training rows. Use the last 20% of available
    # target dates for validation, while preserving chronological order.
    unique_dates = pd.Series(work["dt"].dropna().dt.normalize().unique()).sort_values().reset_index(drop=True)

    if len(unique_dates) < 10:
        raise RuntimeError(
            f"Not enough usable dates for horizon {horizon_label}. "
            f"Only {len(unique_dates)} dates available after target construction."
        )

    n_valid_dates = max(7, int(len(unique_dates) * 0.20))
    n_valid_dates = min(n_valid_dates, len(unique_dates) - 7)

    validation_start = pd.Timestamp(unique_dates.iloc[-n_valid_dates])

    train_df = work[work["dt"] < validation_start].copy()
    valid_df = work[work["dt"] >= validation_start].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        raise RuntimeError(
            f"Bad split for {horizon_label}: train={len(train_df)}, valid={len(valid_df)}, "
            f"available_dates={len(unique_dates)}, validation_start={validation_start.date()}"
        )

    X_train = train_df[FEATURE_COLS].astype(np.float32)
    y_train = train_df[target_col].astype(np.float32)

    X_valid = valid_df[FEATURE_COLS].astype(np.float32)
    y_valid = valid_df[target_col].astype(np.float32)

    print("\n" + "=" * 80)
    print(f"Horizon: {horizon_label} ({horizon_steps} step)")
    print("Train rows:", len(train_df))
    print("Validation rows:", len(valid_df))
    print("Target:", target_col)

    rows = evaluate_baselines(train_df, valid_df, target_col)
    fitted_models = {}

    for model_name, model in candidate_models().items():
        print(f"Training {model_name} for {horizon_label}...")

        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_valid), 0.0)

        row = {
            "model": model_name,
            "model_key": safe_name(model_name),
            "model_class": type(model).__name__,
            "mae": float(mean_absolute_error(y_valid, pred)),
            "rmse": float(root_mean_squared_error(y_valid, pred)),
            "smape": smape(y_valid, pred),
            "is_trainable_model": True,
        }

        print(row)
        rows.append(row)
        fitted_models[model_name] = model

    result = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    result["rank_by_mae"] = np.arange(1, len(result) + 1)
    result["horizon_label"] = horizon_label
    result["horizon_steps"] = horizon_steps
    result["target"] = target_col
    result["train_rows"] = len(train_df)
    result["validation_rows"] = len(valid_df)

    deployable = result[result["is_trainable_model"] == True].copy().sort_values("mae")
    best = deployable.iloc[0]
    best_name = best["model"]
    best_key = best["model_key"]
    best_model = fitted_models[best_name]

    # Refit selected model on all complete rows for this horizon.
    X_all = work[FEATURE_COLS].astype(np.float32)
    y_all = work[target_col].astype(np.float32)
    best_model.fit(X_all, y_all)

    model_path = REGISTRY_DIR / f"{horizon_label}_{best_key}.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "horizon_label": horizon_label,
        "horizon_steps": horizon_steps,
        "selected_model": best_name,
        "selected_model_key": best_key,
        "model_class": type(best_model).__name__,
        "model_path": str(model_path),
        "target": target_col,
        "feature_cols": FEATURE_COLS,
        "categorical_id_cols": ID_COLS,
        "mae": float(best["mae"]),
        "rmse": float(best["rmse"]),
        "smape": float(best["smape"]),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
    }

    return result, metadata


def main():
    df = load_data()

    all_results = []
    registry = {}

    for label, steps in HORIZONS.items():
        result, metadata = train_one_horizon(df, label, steps)
        all_results.append(result)
        registry[label] = metadata

    comparison = pd.concat(all_results, ignore_index=True)
    comparison.to_csv(RESULTS_DIR / "business_horizon_model_comparison.csv", index=False)

    registry_path = MODEL_DIR / "business_horizon_model_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    features_path = MODEL_DIR / "business_horizon_features.pkl"
    joblib.dump(FEATURE_COLS, features_path)

    metadata = {
        "model_family": "business_horizon_forecasting",
        "horizons": HORIZONS,
        "registry_path": str(registry_path),
        "features_path": str(features_path),
        "comparison_path": str(RESULTS_DIR / "business_horizon_model_comparison.csv"),
        "feature_cols": FEATURE_COLS,
        "categorical_id_cols": ID_COLS,
        "note": "Separate horizon-specific models are trained for 1-day, 1-week, and 1-month demand forecasting."
    }

    with open(MODEL_DIR / "business_horizon_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved comparison:", RESULTS_DIR / "business_horizon_model_comparison.csv")
    print("Saved registry:", registry_path)
    print("Saved metadata:", MODEL_DIR / "business_horizon_metadata.json")
    print("\nSelected models:")
    print(json.dumps(registry, indent=2))


if __name__ == "__main__":
    main()
