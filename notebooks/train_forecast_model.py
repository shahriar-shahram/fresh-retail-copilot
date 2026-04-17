import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

df = pd.read_parquet("data/processed/forecast_dataset.parquet")

df = df.sort_values(["store_id", "product_id", "dt"])

# Time split
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

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

X_train = train[feature_cols]
y_train = train["corrected_demand"]

X_test = test[feature_cols]
y_test = test["corrected_demand"]

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
pred = np.clip(pred, 0, None)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

joblib.dump(model, "models/forecast_model.pkl")
print("Saved model: models/forecast_model.pkl")

joblib.dump(feature_cols, "models/forecast_features.pkl")
print("Saved features: models/forecast_features.pkl")

test["prediction"] = pred
test.to_parquet("data/processed/forecast_results.parquet", index=False)

print("Saved forecast results")