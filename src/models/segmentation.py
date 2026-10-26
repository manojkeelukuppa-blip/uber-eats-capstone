
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def prepare_features(df, numeric_cols, categorical_cols):
    """Encode categorical variables and scale numeric variables."""
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()

    return X, preprocessor, feature_names


def run_segmentation(
    dataset_name,
    df,
    id_col,
    numeric_cols,
    categorical_cols,
):
    """Run K-Means, DBSCAN, PCA, and cluster profiling."""
    print(f"\n{'=' * 60}")
    print(f"{dataset_name.upper()} SEGMENTATION")
    print(f"{'=' * 60}")

    # Keep IDs and descriptive columns for interpreting results,
    # but never include them in the model features.
    X, preprocessor, feature_names = prepare_features(
        df, numeric_cols, categorical_cols
    )

    if X.shape[0] < 3:
        raise ValueError(f"{dataset_name}: Need at least 3 rows.")

    print(f"Rows: {X.shape[0]}")
    print(f"Encoded model features: {X.shape[1]}")

    # ---------------- K-MEANS: ELBOW + SILHOUETTE ----------------
    max_k = min(8, len(df) - 1)
    k_values = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    best_index = int(np.argmax(silhouette_scores))
    best_k = k_values[best_index]

    elbow_path = FIGURES_DIR / f"{dataset_name}_elbow.png"
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"{dataset_name.title()} — K-Means Elbow Plot")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=150)
    plt.close()

    silhouette_path = FIGURES_DIR / f"{dataset_name}_silhouette.png"
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title(f"{dataset_name.title()} — K-Means Silhouette Scores")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(silhouette_path, dpi=150)
    plt.close()

    print("\nK-Means evaluation:")
    for k, inertia, score in zip(k_values, inertias, silhouette_scores):
        print(f"k={k}: inertia={inertia:.2f}, silhouette={score:.4f}")

    print(f"Selected k by highest silhouette score: {best_k}")

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    kmeans_labels = kmeans.fit_predict(X)

    # ---------------- DBSCAN: DENSE GROUPS + NOISE ----------------
    # Compare a small range of eps values. Min samples is kept modest
    # because one-hot encoding can make the feature space sparse.
    dbscan_candidates = []
    for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
        candidate = DBSCAN(eps=eps, min_samples=5)
        labels = candidate.fit_predict(X)
        cluster_labels = set(labels) - {-1}
        n_clusters = len(cluster_labels)
        n_noise = int(np.sum(labels == -1))

        score = np.nan
        non_noise_mask = labels != -1
        non_noise_labels = labels[non_noise_mask]

        if (
            n_clusters >= 2
            and len(non_noise_labels) > n_clusters
        ):
            score = silhouette_score(X[non_noise_mask], non_noise_labels)

        dbscan_candidates.append(
            (eps, candidate, labels, n_clusters, n_noise, score)
        )

    # Prefer a result with at least two clusters and a valid silhouette.
    valid_candidates = [
        item for item in dbscan_candidates
        if item[3] >= 2 and not np.isnan(item[5])
    ]

    if valid_candidates:
        selected_dbscan = max(valid_candidates, key=lambda item: item[5])
    else:
        # If DBSCAN cannot find multiple valid groups, retain a transparent
        # fallback candidate with the most clusters and then least noise.
        selected_dbscan = max(
            dbscan_candidates,
            key=lambda item: (item[3], -item[4]),
        )

    eps, dbscan, dbscan_labels, n_db_clusters, n_noise, db_score = (
        selected_dbscan
    )

    print("\nDBSCAN evaluation:")
    for candidate_eps, _, labels, clusters, noise, score in dbscan_candidates:
        score_text = f"{score:.4f}" if not np.isnan(score) else "not defined"
        print(
            f"eps={candidate_eps}: clusters={clusters}, "
            f"noise points={noise}, silhouette={score_text}"
        )

    print(f"Selected DBSCAN eps: {eps}")
    print(f"DBSCAN clusters (excluding noise): {n_db_clusters}")
    print(f"DBSCAN noise points: {n_noise}")

    # ---------------- SAVE MODEL OUTPUTS ----------------
    results = df.copy()
    results["kmeans_cluster"] = kmeans_labels
    results["dbscan_cluster"] = dbscan_labels

    results_path = PROCESSED_DIR / f"{dataset_name}_clustered.csv"
    results.to_csv(results_path, index=False)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": kmeans,
            "feature_names": feature_names.tolist(),
            "selected_k": best_k,
        },
        MODELS_DIR / f"{dataset_name}_kmeans.joblib",
    )

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": dbscan,
            "feature_names": feature_names.tolist(),
            "selected_eps": eps,
        },
        MODELS_DIR / f"{dataset_name}_dbscan.joblib",
    )

    # ---------------- CLUSTER PROFILES ----------------
    profile_cols = [
        c for c in numeric_cols
        if c in results.columns
    ]
    profiles = results.groupby("kmeans_cluster")[profile_cols].mean().round(2)
    profiles["customer_or_restaurant_count"] = (
        results.groupby("kmeans_cluster").size()
    )

    profiles_path = PROCESSED_DIR / f"{dataset_name}_cluster_profiles.csv"
    profiles.to_csv(profiles_path)

    print("\nK-Means cluster profiles:")
    print(profiles.to_string())

    # ---------------- PCA VISUALIZATION ----------------
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    pca_path = FIGURES_DIR / f"{dataset_name}_pca.png"
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        s=20,
        alpha=0.75,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"{dataset_name.title()} — K-Means Clusters (PCA)")
    plt.colorbar(scatter, label="K-Means cluster")
    plt.tight_layout()
    plt.savefig(pca_path, dpi=150)
    plt.close()

    print("\nSaved:")
    print(f"  Clustered data: {results_path.relative_to(ROOT)}")
    print(f"  Cluster profiles: {profiles_path.relative_to(ROOT)}")
    print(f"  Elbow plot: {elbow_path.relative_to(ROOT)}")
    print(f"  Silhouette plot: {silhouette_path.relative_to(ROOT)}")
    print(f"  PCA plot: {pca_path.relative_to(ROOT)}")


def main():
    customers = pd.read_csv(PROCESSED_DIR / "customer_features.csv")
    restaurants = pd.read_csv(PROCESSED_DIR / "restaurant_features.csv")

    # IDs, signup dates, and restaurant names are excluded from modeling.
    # Categorical columns are encoded; numerical columns are standardized.
    run_segmentation(
        dataset_name="customer",
        df=customers,
        id_col="customer_id",
        numeric_cols=[
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
        categorical_cols=[
            "city",
            "membership_type",
            "preferred_cuisine",
        ],
    )

    run_segmentation(
        dataset_name="restaurant",
        df=restaurants,
        id_col="restaurant_id",
        numeric_cols=[
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
        categorical_cols=[
            "city",
            "cuisine",
            "is_active",
        ],
    )


if __name__ == "__main__":
    main()
