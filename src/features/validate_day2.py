
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"


def load_data():
    return {
        file.stem: pd.read_csv(file)
        for file in sorted(DATA_DIR.glob("*.csv"))
    }


def main():
    data = load_data()

    print("=" * 70)
    print("DAY 2 — DATA QUALITY REPORT")
    print("=" * 70)

    for name, df in data.items():
        print(f"\n{name.upper()}")
        print("-" * 50)
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Exact duplicate rows: {df.duplicated().sum():,}")
        print("Missing values:")
        missing = df.isna().sum()
        print(missing[missing > 0].to_string() if missing.any() else "None")

    # Primary-key checks
    primary_keys = {
        "customers": "customer_id",
        "restaurants": "restaurant_id",
        "drivers": "driver_id",
        "orders": "order_id",
        "deliveries": "delivery_id",
        "reviews": "review_id",
        "payments": "payment_id",
    }

    print("\n" + "=" * 70)
    print("PRIMARY-KEY CHECKS")
    print("=" * 70)

    for name, key in primary_keys.items():
        df = data[name]
        duplicates = df[key].duplicated().sum()
        missing = df[key].isna().sum()
        print(
            f"{name}: duplicate {key}={duplicates}, "
            f"missing {key}={missing}"
        )

    # Foreign-key relationship checks
    relationships = [
        ("orders", "customer_id", "customers", "customer_id"),
        ("orders", "restaurant_id", "restaurants", "restaurant_id"),
        ("deliveries", "order_id", "orders", "order_id"),
        ("deliveries", "driver_id", "drivers", "driver_id"),
        ("payments", "order_id", "orders", "order_id"),
        ("reviews", "order_id", "orders", "order_id"),
        ("reviews", "customer_id", "customers", "customer_id"),
        ("reviews", "restaurant_id", "restaurants", "restaurant_id"),
    ]

    print("\n" + "=" * 70)
    print("FOREIGN-KEY CHECKS")
    print("=" * 70)

    for child, child_key, parent, parent_key in relationships:
        child_values = data[child][child_key].dropna()
        parent_values = set(data[parent][parent_key].dropna())
        invalid = (~child_values.isin(parent_values)).sum()

        print(
            f"{child}.{child_key} -> {parent}.{parent_key}: "
            f"{invalid} invalid references"
        )

    # Basic numeric range checks
    print("\n" + "=" * 70)
    print("NUMERIC RANGE CHECKS")
    print("=" * 70)

    checks = {
        "customers": ["age"],
        "orders": [
            "order_value", "delivery_fee",
            "discount_amount", "distance_km", "order_hour"
        ],
        "restaurants": [
            "average_rating", "average_prep_time_min"
        ],
        "deliveries": [
            "prep_time_min", "delivery_time_min"
        ],
        "reviews": ["rating"],
        "payments": ["payment_amount"],
    }

    for table, columns in checks.items():
        for column in columns:
            values = pd.to_numeric(data[table][column], errors="coerce")
            print(
                f"{table}.{column}: "
                f"min={values.min()}, max={values.max()}, "
                f"invalid_numeric={values.isna().sum()}"
            )

    print("\nValidation report completed.")


if __name__ == "__main__":
    main()
