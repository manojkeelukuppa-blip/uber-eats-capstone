
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "customer": {
        "file": "customer_features.csv",
        "eps": 3.0,
        "min_samples": 3,
        "numeric": [
            "age",
            "total_orders",
            "average_order_value",
            "total_spending",
            "ordering_days",
            "weekend_orders",
            "late_night_orders",
            "average_rating_given",
            "review_count",
        ],
        "categorical": [
            "city",
            "membership_type",
            "preferred_cuisine",
        ],
    },
    "restaurant": {
        "file": "restaurant_features.csv",
        "eps": 2.5,
        "min_samples": 5,
        "numeric": [
            "average_rating",
            "average_prep_time_min",
            "total_orders",
            "revenue",
            "average_order_value",
            "cancellation_rate",
            "average_delivery_time",
            "review_average_rating",
            "review_count",
        ],
        "categorical": [
            "city",
            "cuisine",
            "is_active",
        ],
    },
}


def evaluate(name, config):
    # 1. Load feature-engineered data
    input_path = DATA_DIR / config["file"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    # 2. Validate required feature columns
    feature_columns = config["numeric"] + config["categorical"]
    missing_columns = [
        column for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name.title()} dataset is missing columns: "
            f"{missing_columns}"
        )

    if df[feature_columns].isnull().any().any():
        raise ValueError(
            f"{name.title()} features contain missing values. "
            "Resolve them before running DBSCAN."
        )

    # 3. Preprocess numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                config["numeric"],
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                config["categorical"],
            ),
        ]
    )

    X = preprocessor.fit_transform(df[feature_columns])

    # 4. Fit DBSCAN with the selected parameters
    model = DBSCAN(
        eps=config["eps"],
        min_samples=config["min_samples"],
    )

    labels = model.fit_predict(X)

    # 5. Save cluster labels with the original feature data
    output = df.copy()
    output["dbscan_cluster"] = labels

    output_path = DATA_DIR / f"{name}_dbscan_evaluated.csv"
    output.to_csv(output_path, index=False)

    # 6. Save the fitted preprocessor and DBSCAN model together
    model_bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": feature_columns,
        "selected_eps": config["eps"],
        "selected_min_samples": config["min_samples"],
    }

    model_path = MODEL_DIR / f"{name}_dbscan.joblib"
    joblib.dump(model_bundle, model_path)

    # 7. Display evaluation results
    print("\n" + "=" * 60)
    print(f"{name.upper()} DBSCAN RESULTS")
    print("=" * 60)

    print(f"Rows evaluated: {len(output)}")
    print(f"eps: {config['eps']}")
    print(f"min_samples: {config['min_samples']}")

    cluster_sizes = (
        output["dbscan_cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCluster sizes (-1 represents noise):")
    print(cluster_sizes.to_string())

    noise_count = int((labels == -1).sum())
    noise_percentage = (
        noise_count / len(labels) * 100 if len(labels) else 0
    )

    print(f"\nNoise points: {noise_count}")
    print(f"Noise percentage: {noise_percentage:.2f}%")

    print("\nNumeric feature profiles:")
    profiles = (
        output.groupby("dbscan_cluster")[config["numeric"]]
        .mean()
        .round(2)
    )
    print(profiles.to_string())

    print(f"\nSaved evaluated data: {output_path.relative_to(ROOT)}")
    print(f"Saved model bundle: {model_path.relative_to(ROOT)}")


if __name__ == "__main__":
    for dataset_name, dataset_config in CONFIGS.items():
        evaluate(dataset_name, dataset_config)

    print("\nDBSCAN evaluation and model saving completed.")
