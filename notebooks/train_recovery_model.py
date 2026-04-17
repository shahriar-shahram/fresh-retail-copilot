import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# Load data
# =========================
df = pd.read_parquet("data/processed/model_data.parquet")

print("Loaded:", df.shape)

# =========================
# Split data
# =========================
df_train = df[df["in_stock_ratio"] > 0.5].copy()
df_test = df[df["in_stock_ratio"] <= 0.5].copy()

print("Train (clean):", df_train.shape)
print("Test (stockout):", df_test.shape)

# =========================
# Features
# =========================
feature_cols = [
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
    "demand_std"
]

X_train = df_train[feature_cols]
y_train = df_train["sale_amount"]

X_test = df_test[feature_cols]
y_test = df_test["sale_amount"]  # biased but useful for comparison

# =========================
# Train model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# Evaluate on clean data
# =========================
train_pred = model.predict(X_train)

mae = mean_absolute_error(y_train, train_pred)
rmse = np.sqrt(mean_squared_error(y_train, train_pred))

print("\nPerformance on CLEAN data:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# =========================
# Predict latent demand
# =========================
df_test["predicted_true_demand"] = model.predict(X_test)

# Clip negative values
df_test["predicted_true_demand"] = df_test["predicted_true_demand"].clip(lower=0)

print("\nSample latent demand predictions:")
print(df_test[["sale_amount", "predicted_true_demand"]].head(10))

# =========================
# Save results
# =========================
df_test.to_parquet("data/processed/latent_demand_predictions.parquet", index=False)

print("\nSaved latent demand predictions")