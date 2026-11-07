# Register the raw tourism dataset.
# The CSV lives in the GitHub repository, so registration validates it before the pipeline uses it.
# Any failed check raises an error, which stops the GitHub Actions workflow.
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Load the raw dataset from the repository data folder
df = pd.read_csv(RAW_PATH)

# 1. All expected columns must be present
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# 2. Dataset must not be empty
if df.empty:
    raise ValueError("Dataset is empty.")

# 3. Target must be binary (0 / 1)
if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must only contain 0 and 1.")

# 4. CustomerID must be unique
if df["CustomerID"].duplicated().any():
    raise ValueError("CustomerID contains duplicates.")

# Summary
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nMissing values per column:")
print(df.isna().sum()[df.isna().sum() > 0].to_string() or "None")
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts().to_string())
print(f"Purchase rate: {df['ProdTaken'].mean():.2%}")
