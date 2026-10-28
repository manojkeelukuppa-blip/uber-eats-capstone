
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
ORDERS_PATH = ROOT / "data" / "raw" / "orders.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load and validate timestamps
orders = pd.read_csv(ORDERS_PATH)
orders["order_timestamp"] = pd.to_datetime(
    orders["order_timestamp"], errors="coerce"
)
orders = orders.dropna(subset=["order_timestamp"])

if orders.empty:
    raise ValueError("No valid order timestamps were found.")

# Count all incoming orders, including cancelled and failed orders.
orders["hour"] = orders["order_timestamp"].dt.floor("h")

hourly_demand = (
    orders.groupby("hour")
    .size()
    .rename("order_count")
    .to_frame()
)

# Include hours with no orders.
full_hour_range = pd.date_range(
    start=hourly_demand.index.min(),
    end=hourly_demand.index.max(),
    freq="h",
)

hourly_demand = hourly_demand.reindex(
    full_hour_range, fill_value=0
)
hourly_demand.index.name = "timestamp"
hourly_demand = hourly_demand.reset_index()

# Add calendar features for exploratory analysis.
hourly_demand["hour_of_day"] = (
    hourly_demand["timestamp"].dt.hour
)
hourly_demand["day_of_week"] = (
    hourly_demand["timestamp"].dt.day_name()
)
hourly_demand["is_weekend"] = (
    hourly_demand["timestamp"].dt.day_of_week >= 5
)

# Save processed data.
output_path = PROCESSED_DIR / "hourly_demand.csv"
hourly_demand.to_csv(output_path, index=False)

# Summary
print("Hourly demand dataset created.")
print("Shape:", hourly_demand.shape)
print("Date range:", hourly_demand["timestamp"].min(),
      "to", hourly_demand["timestamp"].max())
print("Total hourly counts:", hourly_demand["order_count"].sum())
print("Zero-order hours:", (hourly_demand["order_count"] == 0).sum())
print("Mean orders per hour:", round(hourly_demand["order_count"].mean(), 3))
print("Maximum orders in an hour:", hourly_demand["order_count"].max())

# Plot 1: Hourly demand over time.
plt.figure(figsize=(14, 5))
plt.plot(
    hourly_demand["timestamp"],
    hourly_demand["order_count"],
    linewidth=0.8,
)
plt.title("Hourly Order Demand Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hourly_demand_over_time.png", dpi=150)
plt.close()

# Plot 2: Average demand by hour of day.
hourly_pattern = hourly_demand.groupby(
    "hour_of_day"
)["order_count"].mean()

plt.figure(figsize=(10, 5))
plt.bar(hourly_pattern.index, hourly_pattern.values)
plt.title("Average Order Demand by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Orders")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "demand_by_hour_of_day.png", dpi=150)
plt.close()

# Plot 3: Average demand by day of week.
day_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
daily_pattern = (
    hourly_demand.groupby("day_of_week")["order_count"]
    .mean()
    .reindex(day_order)
)

plt.figure(figsize=(10, 5))
plt.bar(daily_pattern.index, daily_pattern.values)
plt.title("Average Hourly Demand by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Orders per Hour")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "demand_by_day_of_week.png", dpi=150)
plt.close()

print("\nSaved dataset:", output_path)
print("Saved plots in:", FIGURES_DIR)
