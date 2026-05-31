import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== ATTRITION COUNTS ===")
print(df['Attrition'].value_counts())

print("\n=== ATTRITION PERCENTAGE ===")
print(df['Attrition'].value_counts(normalize=True) * 100)

print("\n=== SAMPLE DATA ===")
print(df.head(3))

print("\n=== ANY MISSING VALUES? ===")
print(df.isnull().sum().sum())