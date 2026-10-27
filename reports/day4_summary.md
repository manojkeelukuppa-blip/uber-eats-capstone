# Day 4 — Delivery Time Prediction

## 1. Objective

The objective of Day 4 was to build and evaluate machine learning models that predict food delivery time using the synthetic Uber Eats dataset.

**Target variable:** `delivery_time_min`

## 2. Feature Engineering

The delivery dataset was combined with order, restaurant, and driver information to create a modelling dataset.

- Processed records: 4,538
- Features include delivery distance, preparation time, delivery fee, order value, traffic level, weather, vehicle type, restaurant cuisine, and driver experience.
- Identifier columns were excluded from the model predictors.
- Numeric and categorical features were handled using separate preprocessing pipelines.
- Missing values were handled using median imputation for numeric features and most-frequent imputation for categorical features.
- The processed dataset was saved as `data/processed/delivery_time_features.csv`.

## 3. Exploratory Data Analysis

The delivery-time distribution and feature relationships were examined using charts and correlation analysis.

Key findings:

- Average delivery time: approximately 50.68 minutes.
- Median delivery time: 49 minutes.
- Minimum delivery time: 11 minutes.
- Maximum delivery time: 137 minutes.
- The IQR method identified 59 potential outliers (approximately 1.30% of records). These records were retained because an outlier is not necessarily a data error.
- Distance had the strongest observed correlation with delivery time (approximately 0.654).
- Delivery fee (0.638) and restaurant average preparation time (0.531) were also positively correlated with delivery time.

## 4. Model Development and Evaluation

Four regression models were trained and compared. The data was split into training and testing sets, with 5-fold cross-validation used to compare models by mean absolute error (MAE).

| Model | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Lasso Regression | 3.891 | 3.908 | 5.018 | 0.912 |
| Linear Regression | 3.894 | 3.911 | 5.019 | 0.912 |
| Ridge Regression | 3.895 | 3.912 | 5.020 | 0.912 |
| Random Forest | 4.474 | 4.461 | 5.758 | 0.885 |

Lasso Regression was selected based on the lowest cross-validation MAE.

### Selected Model Performance

- **Test MAE:** approximately 3.91 minutes.
- **Test RMSE:** approximately 5.02 minutes.
- **Test R²:** approximately 0.912.

On the held-out test set, predictions differed from actual delivery times by approximately 3.91 minutes on average.

## 5. Feature Importance

The Random Forest model's feature importance analysis identified the following leading features:

1. Delivery distance
2. Restaurant average preparation time
3. Order preparation time
4. Delivery fee
5. Traffic level
6. Weather

Feature importance describes how much the model relies on a feature; it does not establish a causal relationship.

## 6. Generated Deliverables

- `src/features/delivery_time_features.py`
- `src/features/delivery_time_eda.py`
- `src/features/delivery_time_prediction.py`
- `data/processed/delivery_time_features.csv`
- `reports/day4_model_comparison.csv`
- `reports/day4_feature_importance.csv`
- `models/delivery_time_best_model.joblib`
- EDA plots under `reports/figures/`
- Actual-versus-predicted plot
- Random Forest feature-importance plot

## 7. Business Recommendations

- Use delivery distance and preparation-time estimates as important inputs when estimating delivery duration.
- Monitor traffic and weather conditions when operational data is available and reliable.
- Investigate unusually long deliveries rather than automatically removing them from analysis.
- Compare predicted and actual delivery times regularly to identify systematic estimation errors.
- Validate the model on real, representative delivery data before using predictions for customer-facing ETAs or operational decisions.

## 8. Limitations and Conclusion

The dataset is synthetic, so these results demonstrate a modelling workflow rather than verified real-world Uber Eats performance. The high test R² should not be interpreted as proof of production readiness. Further validation using representative real-world data, time-based testing, and monitoring for data leakage and distribution changes would be required.

Day 4 completed the feature-engineering, exploratory-analysis, regression-modelling, evaluation, and model-comparison stages of the capstone.
