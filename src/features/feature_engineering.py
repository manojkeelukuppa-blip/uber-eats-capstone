
from pathlib import Path
from pyexpat import features
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    restaurants = pd.read_csv(DATA_DIR / "restaurants.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    reviews = pd.read_csv(DATA_DIR / "reviews.csv")
    deliveries = pd.read_csv(DATA_DIR / "deliveries.csv")

    orders["order_timestamp"] = pd.to_datetime (
        orders["order_timestamp"], errors="coerce"
    )
    orders["order_date"] = orders["order_timestamp"].dt.date
    

    # Normalize day names for reliable weekend calculations.
    orders["day_of_week"] = (
        orders["day_of_week"].astype(str).str.strip().str.title()
    )

    return customers, restaurants, orders, reviews, deliveries


def build_customer_features(customers, orders, reviews):
    # Aggregate order behavior to one row per customer.
    customer_orders = orders.groupby("customer_id").agg(
        total_orders=("order_id", "nunique"),
        average_order_value=("order_value", "mean"),
        total_spending=("order_value", "sum"),
        ordering_days=("order_date", "nunique"),
        weekend_orders=(
            "day_of_week",
            lambda s: s.isin(["Saturday", "Sunday"]).sum()
        ),
        late_night_orders=(
            "order_hour",
            lambda s: s.isin([22, 23, 0, 1, 2, 3, 4]).sum()
        ),
    ).reset_index()

    customer_ratings = reviews.groupby("customer_id").agg(
        average_rating_given=("rating", "mean"),
        review_count=("review_id", "nunique"),
    ).reset_index()

    features = customers.merge(
        customer_orders, on="customer_id", how="left"
    ).merge(
        customer_ratings, on="customer_id", how="left"
    )

    count_columns = [
        "total_orders", "ordering_days", "weekend_orders",
        "late_night_orders", "review_count"
    ]
    features[count_columns] = features[count_columns].fillna(0)

    # A customer with no reviews has an unknown rating, not a zero rating.
    features["average_rating_given"] = features[
        "average_rating_given"
    ].fillna(features["average_rating_given"].median())

    features["average_order_value"] = features[
        "average_order_value"
    ].fillna(0)
    features["total_spending"] = features["total_spending"].fillna(0)
    for column in count_columns:
        features[column] = features[column].astype(int)
    return features


def build_restaurant_features(restaurants, orders, reviews, deliveries):
    # Join order data with delivery records.
    order_delivery = orders.merge(
        deliveries[
            ["order_id", "delivery_time_min", "delivery_status"]
        ],
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    restaurant_orders = order_delivery.groupby("restaurant_id").agg(
        total_orders=("order_id", "nunique"),
        revenue=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
        cancellation_rate=(
            "order_status",
            lambda s: s.astype(str).str.lower().eq("cancelled").mean()
        ),
        average_delivery_time=("delivery_time_min", "mean"),
    ).reset_index()

    restaurant_ratings = reviews.groupby("restaurant_id").agg(
        review_average_rating=("rating", "mean"),
        review_count=("review_id", "nunique"),
    ).reset_index()

    features = restaurants.merge(
        restaurant_orders, on="restaurant_id", how="left"
    ).merge(
        restaurant_ratings, on="restaurant_id", how="left"
    )

    # Retain all restaurants, including any with no orders.
    for column in ["total_orders", "revenue", "review_count"]:
        features[column] = features[column].fillna(0)

    features["average_order_value"] = features[
        "average_order_value"
    ].fillna(0)
    features["cancellation_rate"] = features[
        "cancellation_rate"
    ].fillna(0)

    # Use restaurant-level rating if no customer reviews exist.
    features["review_average_rating"] = features[
        "review_average_rating"
    ].fillna(features["average_rating"])

    # Use recorded restaurant prep time as a fallback for missing delivery
    # preparation/duration information only where applicable.
    features["average_delivery_time"] = features[
        "average_delivery_time"
    ].fillna(features["average_prep_time_min"])

    return features


def main():
    customers, restaurants, orders, reviews, deliveries = load_data()

    customer_features = build_customer_features(
        customers, orders, reviews
    )
    restaurant_features = build_restaurant_features(
        restaurants, orders, reviews, deliveries
    )

    customer_path = PROCESSED_DIR / "customer_features.csv"
    restaurant_path = PROCESSED_DIR / "restaurant_features.csv"

    customer_features.to_csv(customer_path, index=False)
    restaurant_features.to_csv(restaurant_path, index=False)

    print("Feature engineering completed.")
    print(f"Customer features: {customer_features.shape}")
    print(f"Restaurant features: {restaurant_features.shape}")
    print(f"Saved: {customer_path.relative_to(ROOT)}")
    print(f"Saved: {restaurant_path.relative_to(ROOT)}")

    print("\nCustomer feature preview:")
    print(customer_features.head().to_string(index=False))

    print("\nRestaurant feature preview:")
    print(restaurant_features.head().to_string(index=False))

    print("\nFeature table checks:")
    print(
        "Unique customer IDs:",
        customer_features["customer_id"].nunique()
    )
    print(
        "Unique restaurant IDs:",
        restaurant_features["restaurant_id"].nunique()
    )


if __name__ == "__main__":
    main()
