import pandas as pd

df = pd.read_parquet("data/interim/mvp_subset.parquet")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nSample rows:")
print(df.head())

print("\nUnique stock_hour6_22_cnt values:")
print(df["stock_hour6_22_cnt"].value_counts().sort_index().head(30))

print("\nSample hours_stock_status values:")
print(df["hours_stock_status"].head(10).tolist())

print("\nSample hours_sale values:")
print(df["hours_sale"].head(10).tolist())

print("\nSale amount summary:")
print(df["sale_amount"].describe())

print("\nDiscount summary:")
print(df["discount"].describe())