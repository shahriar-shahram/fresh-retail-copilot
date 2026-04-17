import pandas as pd

# Load predictions
df = pd.read_parquet("data/processed/latent_demand_predictions.parquet")

# =========================
# Compute lost sales
# =========================
df["lost_sales"] = df["predicted_true_demand"] - df["sale_amount"]

# Clip negative values
df["lost_sales"] = df["lost_sales"].clip(lower=0)

# =========================
# Basic stats
# =========================
total_lost = df["lost_sales"].sum()
avg_lost = df["lost_sales"].mean()

print(f"\nTotal Lost Sales: {total_lost:.2f}")
print(f"Average Lost Sales per row: {avg_lost:.4f}")

# =========================
# Top worst cases
# =========================
top_loss = df.sort_values("lost_sales", ascending=False).head(10)

print("\nTop 10 Lost Sales Cases:")
print(top_loss[[
    "store_id",
    "product_id",
    "dt",
    "sale_amount",
    "predicted_true_demand",
    "lost_sales"
]])

# =========================
# Save
# =========================
df.to_parquet("data/processed/lost_sales.parquet", index=False)

print("\nSaved lost sales dataset")