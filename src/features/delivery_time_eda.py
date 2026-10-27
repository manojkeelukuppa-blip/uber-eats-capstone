
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "delivery_time_features.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def run_delivery_eda():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    target = "delivery_time_min"

    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    print("\n--- Delivery time summary ---")
    print(df[target].describe())

    plt.figure(figsize=(9, 5))
    sns.histplot(df[target], bins=30, kde=True)
    plt.title("Distribution of Delivery Time")
    plt.xlabel("Delivery time (minutes)")
    plt.ylabel("Number of deliveries")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delivery_time_distribution.png", dpi=150)
    plt.close()

    # 2. IQR-based outlier check
    q1 = df[target].quantile(0.25)
    q3 = df[target].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[target] < lower_bound) | (df[target] > upper_bound)
    ]

    print("\n--- IQR outlier check ---")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {len(outliers)}")
    print(f"Outlier percentage: {len(outliers) / len(df) * 100:.2f}%")

    # 3. Correlations among numerical features and target
    numeric_columns = [
        "delivery_time_min",
        "distance_km",
        "order_value",
        "delivery_fee",
        "discount_amount",
        "order_hour",
        "prep_time_min",
        "average_prep_time_min",
        "average_rating",
        "experience_months",
        "driver_rating",
    ]

    correlation = df[numeric_columns].corr()

    print("\n--- Correlation with delivery time ---")
    print(
        correlation[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
        .round(3)
    )

    plt.figure(figsize=(11, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Between Numerical Features and Delivery Time")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delivery_time_correlation.png", dpi=150)
    plt.close()

    # 4. Distance vs delivery time
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="distance_km", y=target, alpha=0.5)
    sns.regplot(
        data=df, x="distance_km", y=target,
        scatter=False, color="red"
    )
    plt.title("Delivery Distance vs Delivery Time")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distance_vs_delivery_time.png", dpi=150)
    plt.close()

    # 5. Preparation time vs delivery time
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="prep_time_min", y=target, alpha=0.5)
    plt.title("Preparation Time vs Delivery Time")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prep_time_vs_delivery_time.png", dpi=150)
    plt.close()

    # 6. Delivery time by traffic level
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="traffic_level", y=target)
    plt.title("Delivery Time by Traffic Level")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "delivery_time_by_traffic.png", dpi=150)
    plt.close()

    print(f"\nEDA charts saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    run_delivery_eda()
