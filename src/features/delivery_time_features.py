
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_delivery_dataset():
    """Merge delivery, order, restaurant, and driver data for regression."""

    deliveries = pd.read_csv(RAW_DIR / "deliveries.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    restaurants = pd.read_csv(RAW_DIR / "restaurants.csv")
    drivers = pd.read_csv(RAW_DIR / "drivers.csv")

    # Keep deliveries with valid numeric targets.
    deliveries["delivery_time_min"] = pd.to_numeric(
        deliveries["delivery_time_min"], errors="coerce"
    )
    deliveries = deliveries.dropna(subset=["delivery_time_min"]).copy()

    # Merge related datasets.
    data = deliveries.merge(
        orders,
        on="order_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_order"),
    )

    data = data.merge(
        restaurants,
        on="restaurant_id",
        how="left",
        validate="many_to_one",
    )

    data = data.merge(
        drivers,
        on="driver_id",
        how="left",
        validate="many_to_one",
    )

    # Remove rows without the information needed for modelling.
    required_columns = [
        "delivery_time_min",
        "distance_km",
        "order_value",
        "delivery_fee",
        "discount_amount",
        "order_hour",
        "day_of_week",
        "traffic_level",
        "weather",
        "prep_time_min",
        "average_prep_time_min",
        "average_rating",
        "vehicle_type",
        "experience_months",
        "driver_rating",
    ]

    data = data.dropna(subset=required_columns).copy()

    # Keep realistic, positive delivery durations.
    data = data[data["delivery_time_min"] > 0].copy()

    # Save the modelling dataset.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "delivery_time_features.csv"
    data.to_csv(output_path, index=False)

    print("Delivery-time feature engineering completed.")
    print(f"Output: {output_path}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print("\nTarget summary:")
    print(data["delivery_time_min"].describe())
    print("\nMissing values in required modelling columns:")
    print(data[required_columns].isna().sum())
    print("\nDelivery statuses:")
    print(data["delivery_status"].value_counts(dropna=False))

    return data


if __name__ == "__main__":
    build_delivery_dataset()
