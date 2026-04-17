import pandas as pd

# Load full training data
df = pd.read_parquet("data/raw/train.parquet")

# Convert time column
df["dt"] = pd.to_datetime(df["dt"])

print("Original shape:", df.shape)

# Pick one city
selected_city = df["city_id"].value_counts().index[0]
df_city = df[df["city_id"] == selected_city].copy()

# Pick top 3 stores in that city by row count
top_stores = df_city["store_id"].value_counts().head(3).index.tolist()
df_city = df_city[df_city["store_id"].isin(top_stores)].copy()

# Pick top 10 products by row count within those stores
top_products = df_city["product_id"].value_counts().head(10).index.tolist()
df_subset = df_city[df_city["product_id"].isin(top_products)].copy()

# Sort
df_subset = df_subset.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)

print("Selected city:", selected_city)
print("Selected stores:", top_stores)
print("Selected products:", top_products)
print("Subset shape:", df_subset.shape)

# Save subset
df_subset.to_parquet("data/interim/mvp_subset.parquet", index=False)

print("Saved to data/interim/mvp_subset.parquet")