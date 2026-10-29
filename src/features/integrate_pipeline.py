
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = {
    "customers": PROCESSED_DIR / "customer_clustered.csv",
    "restaurants": PROCESSED_DIR / "restaurant_clustered.csv",
    "reviews": PROCESSED_DIR / "reviews_with_vader.csv",
    "delivery_features": PROCESSED_DIR / "delivery_time_features.csv",
    "hourly_demand": PROCESSED_DIR / "hourly_demand.csv",
    "demand_forecast": PROCESSED_DIR / "next_24_hour_demand_forecast.csv",
    "delivery_model_comparison": REPORT_DIR / "day4_model_comparison.csv",
    "forecast_model_comparison": REPORT_DIR / "forecast_model_comparison.csv",
    "sentiment_evaluation": REPORT_DIR / "day3_sentiment_evaluation.txt",
    "complaint_analysis": REPORT_DIR / "day3_complaint_analysis.txt",
    "delivery_feature_importance": REPORT_DIR / "day4_feature_importance.csv",
}


def load_csv(name, path):
    """Load an existing CSV and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file for {name!r} was not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"The {name!r} dataset is empty: {path}")

    print(f"Loaded {name}: {len(df):,} rows")
    return df


def read_text(path):
    if not path.exists():
        raise FileNotFoundError(f"Required report was not found: {path}")
    return path.read_text(encoding="utf-8")


def main():
    print("=" * 65)
    print("DAY 6 — END-TO-END ML INTEGRATION")
    print("=" * 65)

    # 1. Load existing analytical outputs without retraining.
    customers = load_csv("customer segments", REQUIRED_FILES["customers"])
    restaurants = load_csv("restaurant segments", REQUIRED_FILES["restaurants"])
    reviews = load_csv("review sentiment", REQUIRED_FILES["reviews"])
    deliveries = load_csv(
        "delivery features", REQUIRED_FILES["delivery_features"]
    )
    hourly = load_csv("hourly demand", REQUIRED_FILES["hourly_demand"])
    forecasts = load_csv(
        "historical demand forecast", REQUIRED_FILES["demand_forecast"]
    )

    delivery_comparison = load_csv(
        "delivery model comparison",
        REQUIRED_FILES["delivery_model_comparison"],
    )
    forecast_comparison = load_csv(
        "forecast model comparison",
        REQUIRED_FILES["forecast_model_comparison"],
    )
    feature_importance = load_csv(
        "delivery feature importance",
        REQUIRED_FILES["delivery_feature_importance"],
    )

    sentiment_evaluation = read_text(REQUIRED_FILES["sentiment_evaluation"])
    complaint_analysis = read_text(REQUIRED_FILES["complaint_analysis"])

    # 2. Validate important columns.
    required_columns = {
        "customers": (customers, {"customer_id", "kmeans_cluster"}),
        "restaurants": (restaurants, {"restaurant_id", "kmeans_cluster"}),
        "reviews": (reviews, {"review_id", "sentiment", "vader_sentiment"}),
        "deliveries": (
            deliveries,
            {"delivery_time_min", "distance_km", "prep_time_min"},
        ),
        "hourly_demand": (hourly, {"timestamp", "order_count"}),
        "forecasts": (forecasts, {"timestamp", "arima_forecast"}),
    }

    for name, (df, columns) in required_columns.items():
        missing = columns - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # 3. Customer segment summary.
    customer_summary = (
        customers.groupby("kmeans_cluster")
        .agg(
            customers=("customer_id", "nunique"),
            average_orders=("total_orders", "mean"),
            average_order_value=("average_order_value", "mean"),
            average_spending=("total_spending", "mean"),
        )
        .reset_index()
    )

    if "cluster_name" in customers.columns:
        names = customers[
            ["kmeans_cluster", "cluster_name"]
        ].drop_duplicates()
        customer_summary = customer_summary.merge(
            names, on="kmeans_cluster", how="left"
        )

    customer_summary.to_csv(
        REPORT_DIR / "day6_customer_segment_summary.csv", index=False
    )

    # 4. Restaurant operational summary.
    restaurant_summary = (
        restaurants.groupby("kmeans_cluster")
        .agg(
            restaurants=("restaurant_id", "nunique"),
            average_orders=("total_orders", "mean"),
            average_revenue=("revenue", "mean"),
            average_prep_time=("average_prep_time_min", "mean"),
            average_delivery_time=("average_delivery_time", "mean"),
            average_cancellation_rate=("cancellation_rate", "mean"),
        )
        .reset_index()
    )

    if "cluster_name" in restaurants.columns:
        names = restaurants[
            ["kmeans_cluster", "cluster_name"]
        ].drop_duplicates()
        restaurant_summary = restaurant_summary.merge(
            names, on="kmeans_cluster", how="left"
        )

    restaurant_summary.to_csv(
        REPORT_DIR / "day6_restaurant_segment_summary.csv", index=False
    )

    # 5. Review sentiment and complaint summaries.
    sentiment_summary = (
        reviews.groupby("sentiment")
        .size()
        .rename("review_count")
        .reset_index()
    )
    sentiment_summary["percentage"] = (
        sentiment_summary["review_count"] / len(reviews) * 100
    )
    sentiment_summary.to_csv(
        REPORT_DIR / "day6_sentiment_summary.csv", index=False
    )

    negative_reviews = reviews[
        reviews["sentiment"].astype(str).str.lower() == "negative"
    ].copy()

    # 6. Delivery metrics and model comparison.
    delivery_times = pd.to_numeric(
        deliveries["delivery_time_min"], errors="coerce"
    ).dropna()

    delivery_summary = {
        "records_with_delivery_features": int(len(deliveries)),
        "valid_delivery_time_records": int(len(delivery_times)),
        "mean_delivery_time_minutes": round(float(delivery_times.mean()), 2),
        "median_delivery_time_minutes": round(float(delivery_times.median()), 2),
        "minimum_delivery_time_minutes": round(float(delivery_times.min()), 2),
        "maximum_delivery_time_minutes": round(float(delivery_times.max()), 2),
    }

    delivery_summary_path = REPORT_DIR / "day6_delivery_summary.json"
    delivery_summary_path.write_text(
        json.dumps(delivery_summary, indent=2), encoding="utf-8"
    )

    # 7. Demand metrics and forecast date range.
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], errors="coerce")
    forecasts["timestamp"] = pd.to_datetime(
        forecasts["timestamp"], errors="coerce"
    )

    counts = pd.to_numeric(hourly["order_count"], errors="coerce").dropna()
    forecast_values = pd.to_numeric(
        forecasts["arima_forecast"], errors="coerce"
    ).dropna()

    demand_summary = {
        "hourly_observations": int(len(hourly)),
        "average_orders_per_hour": round(float(counts.mean()), 3),
        "maximum_orders_in_an_hour": int(counts.max()),
        "historical_data_start": str(hourly["timestamp"].min()),
        "historical_data_end": str(hourly["timestamp"].max()),
        "forecast_start": str(forecasts["timestamp"].min()),
        "forecast_end": str(forecasts["timestamp"].max()),
        "average_arima_forecast": round(float(forecast_values.mean()), 3),
        "forecast_is_historical_not_live": True,
    }

    (REPORT_DIR / "day6_demand_summary.json").write_text(
        json.dumps(demand_summary, indent=2), encoding="utf-8"
    )

    # 8. Build a consolidated report using measured outputs.
    report = [
        "# Day 6 — Integrated ML Pipeline and Business Insights",
        "",
        "## 1. Scope",
        "",
        "This integration consolidates existing Day 2–Day 5 outputs. "
        "It does not retrain models or regenerate synthetic data.",
        "",
        "All datasets are synthetic and are intended for educational use.",
        "",
        "## 2. Customer segmentation",
        "",
        customer_summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        "Interpret clusters using order frequency and spending together. "
        "Cluster IDs are arbitrary labels, not rankings.",
        "",
        "## 3. Restaurant segmentation",
        "",
        restaurant_summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        "Review preparation time, delivery time, cancellation rate, order "
        "volume, and revenue together before prioritizing interventions.",
        "",
        "## 4. Sentiment distribution",
        "",
        sentiment_summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"Negative reviews in the integrated dataset: {len(negative_reviews):,}.",
        "",
        "The sentiment target labels were derived from ratings. "
        "Model performance should not be interpreted as proof of "
        "generalization to unseen real-world review language.",
        "",
        "### Existing sentiment evaluation",
        "",
        "```text",
        sentiment_evaluation.strip(),
        "```",
        "",
        "### Existing complaint analysis",
        "",
        "```text",
        complaint_analysis.strip(),
        "```",
        "",
        "## 5. Delivery-time prediction",
        "",
        "### Delivery summary",
        "",
        pd.DataFrame([delivery_summary]).to_markdown(
            index=False, floatfmt=".2f"
        ),
        "",
        "### Model comparison",
        "",
        delivery_comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Feature importance",
        "",
        feature_importance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "Feature importance indicates predictive association, not causation.",
        "",
        "## 6. Hourly demand forecasting",
        "",
        pd.DataFrame([demand_summary]).to_markdown(
            index=False, floatfmt=".3f"
        ),
        "",
        "### Forecast model comparison",
        "",
        forecast_comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The saved 24-hour forecast is dated July 1, 2025. It is a historical "
        "project output, not a forecast for today's date.",
        "",
        "## 7. Business recommendations",
        "",
        "1. **Customer retention:** Test targeted loyalty benefits for "
        "higher-frequency, higher-spending segments. Evaluate incremental "
        "repeat orders rather than assuming discounts cause retention.",
        "",
        "2. **Restaurant operations:** Review restaurants with longer prep "
        "and delivery times or higher cancellation rates. Investigate "
        "staffing, preparation workflows, and order acceptance before "
        "taking action.",
        "",
        "3. **Customer experience:** Use late-delivery complaints as an "
        "investigation priority, alongside price/value and service issues. "
        "Validate themes with a larger and more varied review corpus.",
        "",
        "4. **Delivery operations:** Use distance, preparation time, fees, "
        "traffic, and weather as predictive signals. Monitor prediction "
        "errors by city, time, and operating conditions.",
        "",
        "5. **Capacity planning:** Use demand forecasts as planning inputs "
        "only after validating them on recent data. Combine predicted demand "
        "with driver availability and service-level targets.",
        "",
        "6. **Model governance:** Track performance over time, compare "
        "against simple baselines, and review errors for under-served "
        "customers, cities, and restaurant groups.",
        "",
        "## 8. Limitations",
        "",
        "- The datasets are synthetic, so conclusions may not transfer "
        "to a real delivery marketplace.",
        "- Customer and restaurant cluster silhouette scores were modest; "
        "the groups are exploratory.",
        "- Sentiment text contains repeated patterns, which can inflate "
        "random-split evaluation scores.",
        "- Demand counts are sparse, and forecast errors remain material.",
        "- The saved demand forecast is historical, not live.",
        "- Predictive relationships do not establish causal effects.",
        "- Production deployment would require monitoring, retraining "
        "policies, privacy controls, and real-world validation.",
        "",
        "## 9. Future improvements",
        "",
        "- Validate with real, consented, representative operational data.",
        "- Use time-based and grouped validation where appropriate.",
        "- Expand review language variety and test on unseen review templates.",
        "- Add rolling-origin demand evaluation and compare with stronger "
        "seasonal baselines.",
        "- Monitor model drift and segment-level errors.",
        "- Connect forecasts to capacity and service-level simulations.",
        "",
        "## 10. Reproducibility",
        "",
        "Run `python src/integrate_pipeline.py` from the project root. "
        "The script reads existing data and reports and writes consolidated "
        "Day 6 summaries under `reports/`.",
        "",
    ]

    output_path = REPORT_DIR / "day6_integrated_business_insights.md"
    output_path.write_text("\n".join(report), encoding="utf-8")

    print("\nDay 6 integration completed successfully.")
    print(f"Integrated report: {output_path}")
    print(f"Customer summary: {REPORT_DIR / 'day6_customer_segment_summary.csv'}")
    print(f"Restaurant summary: {REPORT_DIR / 'day6_restaurant_segment_summary.csv'}")
    print(f"Sentiment summary: {REPORT_DIR / 'day6_sentiment_summary.csv'}")
    print(f"Delivery summary: {delivery_summary_path}")
    print(f"Demand summary: {REPORT_DIR / 'day6_demand_summary.json'}")


if __name__ == "__main__":
    main()
