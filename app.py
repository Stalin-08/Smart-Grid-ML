import pandas as pd

df = pd.read_csv("data/smart_grid_dataset.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())