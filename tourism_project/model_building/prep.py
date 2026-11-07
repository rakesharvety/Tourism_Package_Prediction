"""Data cleaning and train/test split.

Reads the registered dataset from the repository data folder, cleans it,
and writes Xtrain/Xtest/ytrain/ytest CSVs. The GitHub Actions workflow uploads
these four files as an artifact for the model-training job.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

# --- Remove unnecessary columns --------------------------------------------
# 'Unnamed: 0' is a leftover row index from the CSV export and 'CustomerID' is
# a unique identifier. Neither carries predictive signal, and an ID can only
# let a model memorise rows, so both are dropped.
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# --- Clean text columns -----------------------------------------------------
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].str.strip()

# 'Fe Male' is a data-entry variant of 'Female'
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# --- Remove duplicate rows --------------------------------------------------
before = len(df)
df = df.drop_duplicates()
print(f"Duplicate rows removed: {before - len(df)}")

# Missing values are NOT imputed here. Imputation is part of the model pipeline
# (train.py) so it is fitted on training data only, avoiding leakage.
print("Missing values after cleaning:", int(df.isna().sum().sum()))
print("Gender values:", sorted(df["Gender"].unique()))

# --- Split ------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 80/20 split; stratify keeps the (imbalanced) purchase rate equal in both sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Purchase rate - train: {ytrain.mean():.2%}, test: {ytest.mean():.2%}")
print("Data prepared: train/test splits written.")
