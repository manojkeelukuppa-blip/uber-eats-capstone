# Day 6 — Integrated ML Pipeline and Business Insights

## 1. Scope

This integration consolidates existing Day 2–Day 5 outputs. It does not retrain models or regenerate synthetic data.

All datasets are synthetic and are intended for educational use.

## 2. Customer segmentation

|   kmeans_cluster |   customers |   average_orders |   average_order_value |   average_spending | cluster_name                        |
|-----------------:|------------:|-----------------:|----------------------:|-------------------:|:------------------------------------|
|                0 |         629 |             5.01 |                188.48 |             939.71 | Frequent High-Spending Customers    |
|                1 |         871 |             2.12 |                171.55 |             381.94 | Occasional Lower-Spending Customers |

Interpret clusters using order frequency and spending together. Cluster IDs are arbitrary labels, not rankings.

## 3. Restaurant segmentation

|   kmeans_cluster |   restaurants |   average_orders |   average_revenue |   average_prep_time |   average_delivery_time |   average_cancellation_rate | cluster_name                          |
|-----------------:|--------------:|-----------------:|------------------:|--------------------:|------------------------:|----------------------------:|:--------------------------------------|
|                0 |            55 |            45.78 |           8303.77 |               28.38 |                   54.43 |                        0.07 | Higher-Prep-Time Restaurants          |
|                1 |            45 |            55.16 |          10378.60 |               19.31 |                   46.26 |                        0.05 | Higher-Volume Faster-Prep Restaurants |

Review preparation time, delivery time, cancellation rate, order volume, and revenue together before prioritizing interventions.

## 4. Sentiment distribution

| sentiment   |   review_count |   percentage |
|:------------|---------------:|-------------:|
| negative    |            406 |        13.77 |
| neutral     |            502 |        17.02 |
| positive    |           2041 |        69.21 |

Negative reviews in the integrated dataset: 406.

The sentiment target labels were derived from ratings. Model performance should not be interpreted as proof of generalization to unseen real-world review language.

### Existing sentiment evaluation

```text
DAY 3 — SENTIMENT CLASSIFICATION EVALUATION
=======================================================
Total reviews: 2949
Training reviews: 2359
Testing reviews: 590
Model: TF-IDF + Multinomial Naive Bayes
Target: Rating-derived sentiment labels

Accuracy: 1.0000
Macro F1-score: 1.0000

Classification report:
              precision    recall  f1-score   support

    negative       1.00      1.00      1.00        81
     neutral       1.00      1.00      1.00       101
    positive       1.00      1.00      1.00       408

    accuracy                           1.00       590
   macro avg       1.00      1.00      1.00       590
weighted avg       1.00      1.00      1.00       590
```

### Existing complaint analysis

```text
DAY 3 — NEGATIVE REVIEW THEMES
=============================================
Total reviews: 2949
Negative reviews: 406

Complaint theme counts:
complaint_theme
Late or delayed delivery          219
Price and value                    95
Service and overall experience     92

Sample negative reviews by theme:

Late or delayed delivery:
- Food arrived cold and late
- Order was delayed and food was soggy

Price and value:
- Not worth the price

Service and overall experience:
- Poor service and disappointing taste
```

## 5. Delivery-time prediction

### Delivery summary

|   records_with_delivery_features |   valid_delivery_time_records |   mean_delivery_time_minutes |   median_delivery_time_minutes |   minimum_delivery_time_minutes |   maximum_delivery_time_minutes |
|---------------------------------:|------------------------------:|-----------------------------:|-------------------------------:|--------------------------------:|--------------------------------:|
|                          4538.00 |                       4538.00 |                        50.68 |                          49.00 |                           11.00 |                          137.00 |

### Model comparison

| model             |   cv_mae_mean |   cv_mae_std |   test_mae |   test_rmse |   test_r2 |
|:------------------|--------------:|-------------:|-----------:|------------:|----------:|
| Lasso Regression  |        3.8911 |       0.1114 |     3.9076 |      5.0175 |    0.9123 |
| Linear Regression |        3.8944 |       0.1141 |     3.9112 |      5.0186 |    0.9123 |
| Ridge Regression  |        3.8946 |       0.1132 |     3.9123 |      5.0204 |    0.9122 |
| Random Forest     |        4.4738 |       0.1130 |     4.4608 |      5.7584 |    0.8845 |

### Feature importance

| feature                            |   importance |
|:-----------------------------------|-------------:|
| numeric__distance_km               |       0.3417 |
| numeric__average_prep_time_min     |       0.1546 |
| numeric__prep_time_min             |       0.1447 |
| numeric__delivery_fee              |       0.1318 |
| categorical__traffic_level_High    |       0.0682 |
| categorical__weather_Clear         |       0.0545 |
| categorical__traffic_level_Low     |       0.0178 |
| categorical__weather_Storm         |       0.0123 |
| categorical__traffic_level_Medium  |       0.0090 |
| numeric__discount_amount           |       0.0082 |
| numeric__order_value               |       0.0082 |
| numeric__experience_months         |       0.0078 |
| numeric__driver_rating             |       0.0078 |
| numeric__order_hour                |       0.0069 |
| numeric__average_rating            |       0.0063 |
| categorical__weather_Rain          |       0.0053 |
| categorical__vehicle_type_Scooter  |       0.0011 |
| categorical__cuisine_Mexican       |       0.0010 |
| categorical__cuisine_Chinese       |       0.0009 |
| categorical__vehicle_type_Car      |       0.0009 |
| categorical__city_Pune             |       0.0009 |
| categorical__city_Chennai          |       0.0008 |
| categorical__city_Mumbai           |       0.0008 |
| categorical__vehicle_type_Bike     |       0.0008 |
| categorical__city_Hyderabad        |       0.0008 |
| categorical__day_of_week_Saturday  |       0.0007 |
| categorical__day_of_week_Thursday  |       0.0007 |
| categorical__day_of_week_Wednesday |       0.0007 |
| categorical__day_of_week_Sunday    |       0.0007 |
| categorical__day_of_week_Monday    |       0.0007 |
| categorical__day_of_week_Friday    |       0.0007 |
| categorical__day_of_week_Tuesday   |       0.0006 |
| categorical__cuisine_Indian        |       0.0006 |
| categorical__cuisine_Fast Food     |       0.0006 |
| categorical__cuisine_Italian       |       0.0005 |
| categorical__city_Bengaluru        |       0.0005 |

Feature importance indicates predictive association, not causation.

## 6. Hourly demand forecasting

|   hourly_observations |   average_orders_per_hour |   maximum_orders_in_an_hour | historical_data_start   | historical_data_end   | forecast_start      | forecast_end        |   average_arima_forecast | forecast_is_historical_not_live   |
|----------------------:|--------------------------:|----------------------------:|:------------------------|:----------------------|:--------------------|:--------------------|-------------------------:|:----------------------------------|
|                  4343 |                     1.151 |                           6 | 2025-01-01 01:00:00     | 2025-06-30 23:00:00   | 2025-07-01 00:00:00 | 2025-07-01 23:00:00 |                    1.151 | True                              |

### Forecast model comparison

| Model                |    MAE |   RMSE |   WAPE_percent |   MAPE_nonzero_percent |
|:---------------------|-------:|-------:|---------------:|-----------------------:|
| ARIMA(1,0,1)         | 0.8018 | 1.0224 |        70.8118 |                30.5959 |
| Prophet              | 0.8056 | 1.0222 |        71.1480 |                31.1945 |
| Seasonal Naive (24h) | 0.9965 | 1.3217 |        88.0081 |                55.0883 |

The saved 24-hour forecast is dated July 1, 2025. It is a historical project output, not a forecast for today's date.

## 7. Business recommendations

1. **Customer retention:** Test targeted loyalty benefits for higher-frequency, higher-spending segments. Evaluate incremental repeat orders rather than assuming discounts cause retention.

2. **Restaurant operations:** Review restaurants with longer prep and delivery times or higher cancellation rates. Investigate staffing, preparation workflows, and order acceptance before taking action.

3. **Customer experience:** Use late-delivery complaints as an investigation priority, alongside price/value and service issues. Validate themes with a larger and more varied review corpus.

4. **Delivery operations:** Use distance, preparation time, fees, traffic, and weather as predictive signals. Monitor prediction errors by city, time, and operating conditions.

5. **Capacity planning:** Use demand forecasts as planning inputs only after validating them on recent data. Combine predicted demand with driver availability and service-level targets.

6. **Model governance:** Track performance over time, compare against simple baselines, and review errors for under-served customers, cities, and restaurant groups.

## 8. Limitations

- The datasets are synthetic, so conclusions may not transfer to a real delivery marketplace.
- Customer and restaurant cluster silhouette scores were modest; the groups are exploratory.
- Sentiment text contains repeated patterns, which can inflate random-split evaluation scores.
- Demand counts are sparse, and forecast errors remain material.
- The saved demand forecast is historical, not live.
- Predictive relationships do not establish causal effects.
- Production deployment would require monitoring, retraining policies, privacy controls, and real-world validation.

## 9. Future improvements

- Validate with real, consented, representative operational data.
- Use time-based and grouped validation where appropriate.
- Expand review language variety and test on unseen review templates.
- Add rolling-origin demand evaluation and compare with stronger seasonal baselines.
- Monitor model drift and segment-level errors.
- Connect forecasts to capacity and service-level simulations.

## 10. Reproducibility

Run `python src/integrate_pipeline.py` from the project root. The script reads existing data and reports and writes consolidated Day 6 summaries under `reports/`.
