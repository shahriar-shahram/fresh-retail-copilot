import pandas as pd
import numpy as np

# =========================
# Load data
# =========================
df = pd.read_parquet("data/interim/mvp_subset.parquet")

df["dt"] = pd.to_datetime(df["dt"])

print("Loaded:", df.shape)

# =========================
# 1. Stock availability features
# =========================
TOTAL_HOURS = 17  # from 6 to 22

df["in_stock_ratio"] = df["stock_hour6_22_cnt"] / TOTAL_HOURS

df["is_full_stock"] = (df["in_stock_ratio"] > 0.9).astype(int)
df["is_stockout"] = (df["in_stock_ratio"] == 0).astype(int)

# =========================
# 2. Calendar features
# =========================
df["day_of_week"] = df["dt"].dt.dayofweek
df["month"] = df["dt"].dt.month

# =========================
# 3. Hourly array processing
# =========================

def extract_hourly_features(row):
    stock = np.array(row["hours_stock_status"])
    sales = np.array(row["hours_sale"])

    # Safety
    if len(stock) != 24 or len(sales) != 24:
        return pd.Series([0, 0, 0, 0])

    # Demand intensity (sales when in stock)
    available_sales = sales[stock == 1]

    avg_sales_when_available = available_sales.mean() if len(available_sales) > 0 else 0

    # Peak hour
    peak_hour = int(np.argmax(sales))

    # Stockout duration
    stockout_hours = int((stock == 0).sum())

    # Demand concentration
    demand_std = float(np.std(sales))

    return pd.Series([
        avg_sales_when_available,
        peak_hour,
        stockout_hours,
        demand_std
    ])


df[[
    "avg_sales_when_available",
    "peak_hour",
    "stockout_hours",
    "demand_std"
]] = df.apply(extract_hourly_features, axis=1)

# =========================
# 4. Sort for time series
# =========================
df = df.sort_values(["store_id", "product_id", "dt"])

# =========================
# 5. Lag features
# =========================
df["lag_1"] = df.groupby(["store_id", "product_id"])["sale_amount"].shift(1)
df["lag_7"] = df.groupby(["store_id", "product_id"])["sale_amount"].shift(7)

# =========================
# 6. Rolling features
# =========================
df["rolling_mean_7"] = (
    df.groupby(["store_id", "product_id"])["sale_amount"]
    .shift(1)
    .rolling(7)
    .mean()
)

# =========================
# 7. Clean
# =========================
df_model = df.dropna().reset_index(drop=True)

print("Final model shape:", df_model.shape)

# =========================
# Save
# =========================
df_model.to_parquet("data/processed/model_data.parquet", index=False)

print("Saved: data/processed/model_data.parquet")