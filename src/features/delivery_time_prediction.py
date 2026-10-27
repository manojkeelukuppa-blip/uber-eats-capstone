
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "delivery_time_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    target = "delivery_time_min"

    # Exclude identifiers, target, status, and unrelated text fields.
    numeric_features = [
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

    categorical_features = [
        "day_of_week",
        "traffic_level",
        "weather",
        "vehicle_type",
        "cuisine",
        "city",
    ]

    features = numeric_features + categorical_features
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator),
        ])

        # Negative MAE is used because sklearn maximizes scores.
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "cv_mae_mean": -cv_scores.mean(),
            "cv_mae_std": cv_scores.std(),
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2,
        })

        fitted_models[name] = pipeline

        print(f"\n{name}")
        print(f"  CV MAE: {(-cv_scores.mean()):.3f} +/- {cv_scores.std():.3f}")
        print(f"  Test MAE: {mae:.3f}")
        print(f"  Test RMSE: {rmse:.3f}")
        print(f"  Test R2: {r2:.3f}")

    results_df = pd.DataFrame(results).sort_values("test_mae")
    results_df.to_csv(REPORT_DIR / "day4_model_comparison.csv", index=False)

    print("\n--- Model comparison ---")
    print(results_df.round(3).to_string(index=False))

    # Select the model with the lowest held-out test MAE.
    
# Select the model using cross-validation, not test-set performance.
    best_name = results_df.loc[
    results_df["cv_mae_mean"].idxmin(), "model"
]
    best_model = fitted_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "delivery_time_best_model.joblib")

    print(f"\nSelected model by CV MAE: {best_name}")


    # Actual vs predicted plot for the selected model.
    best_predictions = best_model.predict(X_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_predictions, alpha=0.5)
    lower = min(y_test.min(), best_predictions.min())
    upper = max(y_test.max(), best_predictions.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual delivery time (minutes)")
    plt.ylabel("Predicted delivery time (minutes)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "delivery_time_actual_vs_predicted.png", dpi=150)
    plt.close()

    # Feature importance for Random Forest.
    rf_pipeline = fitted_models["Random Forest"]
    rf_preprocess = rf_pipeline.named_steps["preprocess"]
    rf_model = rf_pipeline.named_steps["model"]
    feature_names = rf_preprocess.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(REPORT_DIR / "day4_feature_importance.csv", index=False)

    top_features = importance.head(15).sort_values("importance")

    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.xlabel("Feature importance")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "delivery_time_feature_importance.png", dpi=150)
    plt.close()

    print("\nTop 10 Random Forest features:")
    print(importance.head(10).round(4).to_string(index=False))

    print("\nSaved model comparison, feature importance, model, and plots.")


if __name__ == "__main__":
    main()
