
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

DATASETS = {
    "customer": {
        "file": "customer_features.csv",
        "numeric": [
            "age", "total_orders", "average_order_value",
            "total_spending", "ordering_days", "weekend_orders",
            "late_night_orders", "average_rating_given", "review_count",
        ],
        "categorical": ["city", "membership_type", "preferred_cuisine"],
    },
    "restaurant": {
        "file": "restaurant_features.csv",
        "numeric": [
            "average_rating", "average_prep_time_min", "total_orders",
            "revenue", "average_order_value", "cancellation_rate",
            "average_delivery_time", "review_average_rating", "review_count",
        ],
        "categorical": ["city", "cuisine", "is_active"],
    },
}


def tune_dataset(name, config):
    df = pd.read_csv(DATA_DIR / config["file"])
    numeric = config["numeric"]
    categorical = config["categorical"]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        ), categorical),
    ])

    X = preprocessor.fit_transform(df)

    # DBSCAN is sensitive to eps. Test a range rather than assuming
    # one setting is appropriate for every dataset.
    eps_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    min_samples_values = [3, 5, 8, 10]

    rows = []

    for min_samples in min_samples_values:
        for eps in eps_values:
            labels = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            ).fit_predict(X)

            non_noise = labels != -1
            cluster_ids = set(labels[non_noise])
            n_clusters = len(cluster_ids)
            noise_count = int((labels == -1).sum())
            score = np.nan

            # Silhouette is meaningful only with at least two clusters.
            if n_clusters >= 2 and non_noise.sum() > n_clusters:
                score = silhouette_score(
                    X[non_noise], labels[non_noise]
                )

            rows.append({
                "dataset": name,
                "eps": eps,
                "min_samples": min_samples,
                "clusters": n_clusters,
                "noise_count": noise_count,
                "noise_percent": round(noise_count / len(df) * 100, 2),
                "silhouette_non_noise": score,
            })

    results = pd.DataFrame(rows)
    output = DATA_DIR / f"{name}_dbscan_tuning.csv"
    results.to_csv(output, index=False)

    print(f"\n{name.upper()} — DBSCAN parameter comparison")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {output.relative_to(ROOT)}")

    valid = results.dropna(subset=["silhouette_non_noise"])
    valid = valid[valid["clusters"] >= 2]

    if not valid.empty:
        # Prefer silhouette, but inspect noise percentage and cluster sizes
        # before choosing a final configuration.
        print("\nTop candidates by non-noise silhouette:")
        print(
            valid.sort_values("silhouette_non_noise", ascending=False)
            .head(5)
            .to_string(index=False)
        )
    else:
        print(
            "\nNo tested configuration produced a valid silhouette "
            "score with at least two non-noise clusters."
        )


if __name__ == "__main__":
    for dataset_name, settings in DATASETS.items():
        tune_dataset(dataset_name, settings)
