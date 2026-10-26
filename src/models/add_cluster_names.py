
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Business-friendly names based on the current K-Means cluster profiles.
cluster_names = {
    "customer": {
        0: "Frequent High-Spending Customers",
        1: "Occasional Lower-Spending Customers",
    },
    "restaurant": {
        0: "Higher-Prep-Time Restaurants",
        1: "Higher-Volume Faster-Prep Restaurants",
    },
}

for dataset_name, name_mapping in cluster_names.items():
    file_path = PROCESSED_DIR / f"{dataset_name}_clustered.csv"

    df = pd.read_csv(file_path)

    if "kmeans_cluster" not in df.columns:
        raise ValueError(
            f"'kmeans_cluster' column not found in {file_path}"
        )

    df["cluster_name"] = df["kmeans_cluster"].map(name_mapping)

    if df["cluster_name"].isna().any():
        missing_clusters = sorted(
            df.loc[df["cluster_name"].isna(), "kmeans_cluster"].unique()
        )
        raise ValueError(
            f"Missing business names for {dataset_name} clusters: "
            f"{missing_clusters}"
        )

    df.to_csv(file_path, index=False)

    print(f"\n{dataset_name.title()} cluster names added:")
    print(
        df[["kmeans_cluster", "cluster_name"]]
        .drop_duplicates()
        .sort_values("kmeans_cluster")
        .to_string(index=False)
    )

print("\nCluster naming completed successfully.")
