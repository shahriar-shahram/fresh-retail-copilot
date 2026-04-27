from pathlib import Path

import numpy as np
import pandas as pd


INPUT_TRAIN = Path("data/processed/model_train_full.parquet")
INPUT_EVAL = Path("data/processed/model_eval_full.parquet")

OUT_TRAIN = Path("data/processed/model_train_rich.parquet")
OUT_EVAL = Path("data/processed/model_eval_rich.parquet")
OUT_APP_SAMPLE = Path("data/processed/app_forecast_scenarios_rich.parquet")

OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

ID_COLS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]

BASE_CONTEXT_COLS = [
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "stock_hour6_22_cnt",
    "in_stock_ratio",
    "stockout_hours",
    "avg_sales_when_available",
]

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [7, 14, 28]


def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ID_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype("int32")

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    df["day_of_week"] = df["dt"].dt.dayofweek.astype("int16")
    df["month"] = df["dt"].dt.month.astype("int16")
    df["day_of_month"] = df["dt"].dt.day.astype("int16")
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype("int16")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)
    g = df.groupby(["store_id", "product_id"], group_keys=False)

    for lag in LAGS:
        df[f"lag_{lag}"] = g["sale_amount"].shift(lag)

    shifted = g["sale_amount"].shift(1)

    for window in ROLL_WINDOWS:
        grouped_shifted = shifted.groupby([df["store_id"], df["product_id"]])

        df[f"rolling_mean_{window}"] = (
            grouped_shifted
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"rolling_std_{window}"] = (
            grouped_shifted
            .rolling(window, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"rolling_min_{window}"] = (
            grouped_shifted
            .rolling(window, min_periods=1)
            .min()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"rolling_max_{window}"] = (
            grouped_shifted
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=[0, 1], drop=True)
        )

    # Conservative fills for early rows in each store-product sequence.
    df["lag_1"] = df["lag_1"].fillna(df["sale_amount"])

    for lag in LAGS:
        df[f"lag_{lag}"] = df[f"lag_{lag}"].fillna(df["lag_1"])

    for window in ROLL_WINDOWS:
        df[f"rolling_mean_{window}"] = df[f"rolling_mean_{window}"].fillna(df["lag_1"])
        df[f"rolling_std_{window}"] = df[f"rolling_std_{window}"].fillna(0.0)
        df[f"rolling_min_{window}"] = df[f"rolling_min_{window}"].fillna(df["lag_1"])
        df[f"rolling_max_{window}"] = df[f"rolling_max_{window}"].fillna(df["lag_1"])

    return df


def build_features(path: Path, split_name: str) -> pd.DataFrame:
    print(f"Loading {split_name}: {path}")
    df = pd.read_parquet(path)
    print(f"{split_name} input shape:", df.shape)

    df["split"] = split_name
    df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce")

    df = normalize_ids(df)
    df = add_calendar_features(df)
    df = add_lag_rolling_features(df)

    for col in BASE_CONTEXT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = (
        ID_COLS
        + BASE_CONTEXT_COLS
        + ["sale_amount", "day_of_week", "month", "day_of_month", "week_of_year", "is_weekend"]
        + [f"lag_{lag}" for lag in LAGS]
        + [f"rolling_mean_{w}" for w in ROLL_WINDOWS]
        + [f"rolling_std_{w}" for w in ROLL_WINDOWS]
        + [f"rolling_min_{w}" for w in ROLL_WINDOWS]
        + [f"rolling_max_{w}" for w in ROLL_WINDOWS]
    )

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=["dt", "sale_amount"] + numeric_cols).reset_index(drop=True)
    print(f"{split_name} output shape:", df.shape)

    return df


def main():
    train = build_features(INPUT_TRAIN, "train")
    eval_df = build_features(INPUT_EVAL, "eval")

    train.to_parquet(OUT_TRAIN, index=False)
    eval_df.to_parquet(OUT_EVAL, index=False)

    # Lightweight file for frontend scenario selection.
    app_sample = (
        pd.concat([train.tail(200_000), eval_df.tail(50_000)], ignore_index=True)
        .sort_values(["store_id", "product_id", "dt"])
        .reset_index(drop=True)
    )
    app_sample.to_parquet(OUT_APP_SAMPLE, index=False)

    print("Saved:", OUT_TRAIN, train.shape)
    print("Saved:", OUT_EVAL, eval_df.shape)
    print("Saved:", OUT_APP_SAMPLE, app_sample.shape)


if __name__ == "__main__":
    main()
