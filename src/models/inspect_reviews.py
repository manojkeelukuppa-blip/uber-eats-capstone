
from pathlib import Path

import pandas as pd

# Locate the project and review dataset
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "reviews.csv"

if not REVIEWS_PATH.exists():
    raise FileNotFoundError(
        f"Reviews dataset not found at: {REVIEWS_PATH}"
    )

# Load the data
reviews = pd.read_csv(REVIEWS_PATH)

print("=" * 60)
print("DAY 3: CUSTOMER REVIEW DATA INSPECTION")
print("=" * 60)

print("\n1. Dataset shape:")
print(reviews.shape)

print("\n2. Column names:")
print(reviews.columns.tolist())

print("\n3. Data types:")
print(reviews.dtypes)

print("\n4. First five reviews:")
print(reviews.head().to_string())

print("\n5. Missing values:")
print(reviews.isnull().sum())

print("\n6. Duplicate rows:")
print(reviews.duplicated().sum())

print("\n7. Unique values per column:")
print(reviews.nunique())

# Inspect likely rating and review-text columns
rating_columns = [
    col for col in reviews.columns
    if "rating" in col.lower() or "sentiment" in col.lower()
]

text_columns = [
    col for col in reviews.columns
    if any(word in col.lower()
           for word in ["review", "comment", "feedback", "text"])
]

print("\n8. Potential rating/sentiment columns:")
for col in rating_columns:
    print(f"\n{col}:")
    print(reviews[col].value_counts(dropna=False).sort_index())

print("\n9. Potential review-text columns:")
for col in text_columns:
    print(f"\n{col} — sample values:")
    print(reviews[col].dropna().head(10).to_list())

print("\nInspection complete.")
