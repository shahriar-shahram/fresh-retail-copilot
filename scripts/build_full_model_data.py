from pathlib import Path
import numpy as np
import pandas as pd


RAW_TRAIN = Path("data/raw/train.parquet")
RAW_EVAL = Path("data/raw/eval.parquet")

OUT_TRAIN = Path("data/processed/model_train_full.parquet")
OUT_EVAL = Path("data/processed/model_eval_full.parquet")
OUT_COMBINED = Path("data/processed/model_data_full.parquet")

OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
    "day_of_week",
    "month",
    "avg_sales_when_available",
    "stockout_hours",
    "demand_std",
]


def prepare_base(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    df["split"] = split

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["sale_amount"] = pd.to_numeric(df["sale_amount"], errors="coerce")

    id_cols = [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]

    for col in id_cols:
        df[col] = df[col].astype(str)

    numeric_cols = [
        "stock_hour6_22_cnt",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # In this dataset, stock_hour6_22_cnt counts available hours between hour 6 and hour 22.
    # That window has 17 hourly positions, so we normalize by 17.
    df["in_stock_ratio"] = (df["stock_hour6_22_cnt"] / 17.0).clip(0, 1)
    df["stockout_hours"] = (17.0 - df["stock_hour6_22_cnt"]).clip(lower=0)

    keep_cols = [
        "split",
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt",
        "in_stock_ratio",
        "stockout_hours",
        "discount",
        "holiday_flag",
        "activity_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]

    df = df[keep_cols]
    df = df.dropna(subset=["dt", "sale_amount"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)

    group_cols = ["store_id", "product_id"]
    g = df.groupby(group_cols, group_keys=False)

    df["lag_1"] = g["sale_amount"].shift(1)
    df["lag_7"] = g["sale_amount"].shift(7)

    shifted_sales = g["sale_amount"].shift(1)

    df["rolling_mean_7"] = (
        shifted_sales
        .groupby([df["store_id"], df["product_id"]])
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["demand_std"] = (
        shifted_sales
        .groupby([df["store_id"], df["product_id"]])
        .rolling(7, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    available_sales = df["sale_amount"].where(df["in_stock_ratio"] > 0.5)

    df["avg_sales_when_available"] = (
        available_sales
        .groupby([df["store_id"], df["product_id"]])
        .transform("mean")
    )

    df["day_of_week"] = df["dt"].dt.dayofweek
    df["month"] = df["dt"].dt.month

    # Conservative fills for early rows per store-product sequence.
    df["lag_1"] = df["lag_1"].fillna(df["sale_amount"])
    df["lag_7"] = df["lag_7"].fillna(df["lag_1"])
    df["rolling_mean_7"] = df["rolling_mean_7"].fillna(df["lag_1"])
    df["demand_std"] = df["demand_std"].fillna(0.0)
    df["avg_sales_when_available"] = df["avg_sales_when_available"].fillna(df["rolling_mean_7"])

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean only numeric modeling columns. Some raw columns contain arrays/lists
    # such as hours_sale and hours_stock_status, so replacing across the full
    # dataframe can fail.
    numeric_model_cols = FEATURE_COLS + ["sale_amount"]

    for col in numeric_model_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=numeric_model_cols).reset_index(drop=True)

    return df


def main():
    print("Loading raw train...")
    train = pd.read_parquet(RAW_TRAIN)
    print("Raw train shape:", train.shape)

    print("Loading raw eval...")
    eval_df = pd.read_parquet(RAW_EVAL)
    print("Raw eval shape:", eval_df.shape)

    train = prepare_base(train, "train")
    eval_df = prepare_base(eval_df, "eval")

    combined = pd.concat([train, eval_df], ignore_index=True)
    print("Combined base shape:", combined.shape)

    print("Building full lag/rolling features...")
    combined = add_time_features(combined)

    model_train = combined[combined["split"] == "train"].copy()
    model_eval = combined[combined["split"] == "eval"].copy()

    print("Final full train shape:", model_train.shape)
    print("Final full eval shape:", model_eval.shape)
    print("Final combined shape:", combined.shape)

    model_train.to_parquet(OUT_TRAIN, index=False)
    model_eval.to_parquet(OUT_EVAL, index=False)
    combined.to_parquet(OUT_COMBINED, index=False)

    print(f"Saved {OUT_TRAIN}")
    print(f"Saved {OUT_EVAL}")
    print(f"Saved {OUT_COMBINED}")

    print("\nFeature columns:")
    for col in FEATURE_COLS:
        print("-", col)


if __name__ == "__main__":
    main()
