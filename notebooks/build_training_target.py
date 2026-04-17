import pandas as pd

df = pd.read_parquet("data/processed/model_data.parquet")
pred = pd.read_parquet("data/processed/latent_demand_predictions.parquet")

# Merge predictions back (only for rows we predicted)
merge_cols = ["store_id", "product_id", "dt"]
df = df.merge(
    pred[merge_cols + ["predicted_true_demand"]],
    on=merge_cols,
    how="left"
)

# Build corrected target
# If high availability → use observed
# Else → use predicted (recovered)
df["corrected_demand"] = df["sale_amount"]

mask = df["in_stock_ratio"] <= 0.5
df.loc[mask, "corrected_demand"] = df.loc[mask, "predicted_true_demand"]

# Clean
df = df.dropna(subset=["corrected_demand"]).reset_index(drop=True)

print("Final dataset:", df.shape)

df.to_parquet("data/processed/forecast_dataset.parquet", index=False)
print("Saved: data/processed/forecast_dataset.parquet")