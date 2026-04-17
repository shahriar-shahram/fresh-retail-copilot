import os
import pandas as pd

RAW_DIR = "data/raw"

print("Files in raw directory:")
for root, dirs, files in os.walk(RAW_DIR):
    for f in files:
        print(os.path.join(root, f))

candidate_files = []
for root, dirs, files in os.walk(RAW_DIR):
    for f in files:
        if f.endswith(".parquet") or f.endswith(".csv"):
            candidate_files.append(os.path.join(root, f))

if not candidate_files:
    raise ValueError("No data files found in data/raw")

file_path = candidate_files[0]
print("\nReading:", file_path)

if file_path.endswith(".parquet"):
    df = pd.read_parquet(file_path)
else:
    df = pd.read_csv(file_path)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)

print("\nHead:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))