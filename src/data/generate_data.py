
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
rng = np.random.default_rng(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_customers(n=1500):
    customer_ids = np.arange(1, n + 1)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "age": rng.integers(18, 71, n),
        "city": rng.choice(
            ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Pune"],
            n,
            p=[0.28, 0.18, 0.20, 0.22, 0.12],
        ),
        "signup_date": pd.to_datetime(
            rng.integers(
                pd.Timestamp("2024-01-01").value // 10**9,
                pd.Timestamp("2025-07-01").value // 10**9,
                n,
            ),
            unit="s",
        ).date,
        "membership_type": rng.choice(
            ["Basic", "Plus", "Premium"], n, p=[0.55, 0.30, 0.15]
        ),
        "preferred_cuisine": rng.choice(
            ["Indian", "Chinese", "Italian", "Mexican", "Fast Food"],
            n,
        ),
    })


def generate_restaurants(n=100):
    return pd.DataFrame({
        "restaurant_id": np.arange(1, n + 1),
        "restaurant_name": [f"Restaurant_{i:03d}" for i in range(1, n + 1)],
        "city": rng.choice(
            ["Bengaluru", "Chennai", "Hyderabad", "Mumbai", "Pune"], n
        ),
        "cuisine": rng.choice(
            ["Indian", "Chinese", "Italian", "Mexican", "Fast Food"], n
        ),
        "average_rating": np.round(rng.uniform(3.0, 5.0, n), 2),
        "average_prep_time_min": rng.integers(10, 41, n),
        "is_active": rng.choice([True, False], n, p=[0.94, 0.06]),
    })


def generate_drivers(n=200):
    return pd.DataFrame({
        "driver_id": np.arange(1, n + 1),
        "vehicle_type": rng.choice(
            ["Bike", "Scooter", "Car"], n, p=[0.25, 0.60, 0.15]
        ),
        "experience_months": rng.integers(1, 97, n),
        "driver_rating": np.round(rng.uniform(3.2, 5.0, n), 2),
    })


def generate_orders(customers, restaurants, n=5000):
    order_ids = np.arange(1, n + 1)
    start = pd.Timestamp("2025-01-01")
    end = pd.Timestamp("2025-07-01")

    start_seconds = int(start.timestamp())
    end_seconds = int(end.timestamp())

    order_times = pd.to_datetime(
        rng.integers(start_seconds, end_seconds, n), unit="s"
    ).floor("s")

    customer_ids = rng.choice(customers["customer_id"], n)
    restaurant_ids = rng.choice(restaurants["restaurant_id"], n)

    subtotal = np.round(rng.lognormal(mean=5.1, sigma=0.45, size=n), 2)
    subtotal = np.clip(subtotal, 80, 1800)

    distance = np.round(rng.gamma(shape=2.0, scale=2.0, size=n), 2)
    distance = np.clip(distance, 0.5, 18)

    delivery_fee = np.round(20 + distance * 7 + rng.uniform(0, 15, n), 2)
    discount = np.round(
        np.minimum(subtotal * rng.uniform(0, 0.18, n), 200), 2
    )

    order_status = rng.choice(
        ["Delivered", "Cancelled", "Failed"],
        n,
        p=[0.91, 0.06, 0.03],
    )

    orders = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "restaurant_id": restaurant_ids,
        "order_timestamp": order_times,
        "order_hour": order_times.hour,
        "day_of_week": order_times.day_name(),
        "order_value": subtotal,
        "delivery_fee": delivery_fee,
        "discount_amount": discount,
        "distance_km": distance,
        "order_status": order_status,
    })

    return orders


def generate_deliveries(orders, restaurants, drivers):
    delivered_orders = orders.loc[
        orders["order_status"] == "Delivered"
    ].copy()

    prep_lookup = restaurants.set_index("restaurant_id")[
        "average_prep_time_min"
    ]

    prep_time = delivered_orders["restaurant_id"].map(prep_lookup).to_numpy()
    traffic = rng.choice(
        ["Low", "Medium", "High"], len(delivered_orders), p=[0.35, 0.45, 0.20]
    )
    weather = rng.choice(
        ["Clear", "Rain", "Storm"], len(delivered_orders), p=[0.75, 0.20, 0.05]
    )

    traffic_delay = np.select(
        [traffic == "Low", traffic == "Medium", traffic == "High"],
        [2, 8, 18],
    )
    weather_delay = np.select(
        [weather == "Clear", weather == "Rain", weather == "Storm"],
        [0, 8, 20],
    )

    distance = delivered_orders["distance_km"].to_numpy()
    travel_time = distance * rng.uniform(3.0, 5.0, len(delivered_orders))

    delivery_minutes = np.maximum(
        10,
        prep_time + travel_time + traffic_delay + weather_delay
        + rng.normal(0, 4, len(delivered_orders)),
    ).round().astype(int)

    delivery = pd.DataFrame({
        "delivery_id": np.arange(1, len(delivered_orders) + 1),
        "order_id": delivered_orders["order_id"].to_numpy(),
        "driver_id": rng.choice(drivers["driver_id"], len(delivered_orders)),
        "traffic_level": traffic,
        "weather": weather,
        "prep_time_min": prep_time,
        "delivery_time_min": delivery_minutes,
        "delivery_status": "Delivered",
    })

    return delivery


def generate_reviews(orders, restaurants):
    delivered_orders = orders.loc[
        orders["order_status"] == "Delivered"
    ].copy()

    review_count = int(len(delivered_orders) * 0.65)
    reviewed = delivered_orders.sample(
        n=review_count, random_state=SEED
    ).copy()

    ratings = rng.choice(
        [1, 2, 3, 4, 5],
        review_count,
        p=[0.06, 0.09, 0.17, 0.30, 0.38],
    )

    positive_text = [
        "Delicious food and quick delivery",
        "Great taste and excellent service",
        "Fresh food, highly recommended",
        "Amazing experience, arrived on time",
    ]
    neutral_text = [
        "Food was okay",
        "Average experience",
        "Taste and delivery were acceptable",
        "Nothing special, but fine",
    ]
    negative_text = [
        "Food arrived cold and late",
        "Poor service and disappointing taste",
        "Order was delayed and food was soggy",
        "Not worth the price",
    ]

    review_text = []
    for rating in ratings:
        if rating >= 4:
            review_text.append(rng.choice(positive_text))
        elif rating == 3:
            review_text.append(rng.choice(neutral_text))
        else:
            review_text.append(rng.choice(negative_text))

    return pd.DataFrame({
        "review_id": np.arange(1, review_count + 1),
        "order_id": reviewed["order_id"].to_numpy(),
        "customer_id": reviewed["customer_id"].to_numpy(),
        "restaurant_id": reviewed["restaurant_id"].to_numpy(),
        "rating": ratings,
        "review_text": review_text,
        "review_date": reviewed["order_timestamp"].dt.date.to_numpy(),
    })


def generate_payments(orders):
    payment_status = np.where(
        orders["order_status"].eq("Delivered"),
        "Successful",
        rng.choice(["Successful", "Failed", "Refunded"], len(orders),
                   p=[0.45, 0.30, 0.25]),
    )

    amount = (
        orders["order_value"]
        + orders["delivery_fee"]
        - orders["discount_amount"]
    ).clip(lower=0)

    return pd.DataFrame({
        "payment_id": np.arange(1, len(orders) + 1),
        "order_id": orders["order_id"].to_numpy(),
        "payment_method": rng.choice(
            ["UPI", "Card", "Wallet", "Cash"], len(orders),
            p=[0.45, 0.25, 0.15, 0.15],
        ),
        "payment_amount": np.round(amount, 2),
        "payment_status": payment_status,
    })


def main():
    customers = generate_customers()
    restaurants = generate_restaurants()
    drivers = generate_drivers()
    orders = generate_orders(customers, restaurants)
    deliveries = generate_deliveries(orders, restaurants, drivers)
    reviews = generate_reviews(orders, restaurants)
    payments = generate_payments(orders)

    datasets = {
        "customers.csv": customers,
        "restaurants.csv": restaurants,
        "drivers.csv": drivers,
        "orders.csv": orders,
        "deliveries.csv": deliveries,
        "reviews.csv": reviews,
        "payments.csv": payments,
    }

    for filename, dataframe in datasets.items():
        output_path = RAW_DATA_DIR / filename
        dataframe.to_csv(output_path, index=False)
        print(f"Created {output_path.relative_to(PROJECT_ROOT)}: "
              f"{len(dataframe):,} rows")

    print("\nSynthetic data generation completed successfully.")


if __name__ == "__main__":
    main()
