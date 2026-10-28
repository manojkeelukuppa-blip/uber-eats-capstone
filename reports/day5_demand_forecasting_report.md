# Day 5: Hourly Demand Forecasting

## 1. Objective

The objective was to analyze hourly order demand, investigate trend and seasonality, test stationarity, compare ARIMA and Prophet forecasting models, and generate a 24-hour forecast.

## 2. Data Preparation

The synthetic orders dataset contained 5,000 orders spanning January 1 through June 30, 2025. Order timestamps were converted to datetime, grouped into hourly counts, and reindexed to include hours with zero orders.

The resulting dataset contained 4,343 hourly observations, with an average of 1.151 orders per hour and a maximum of 6 orders in an hour. All order statuses were included because the analysis focuses on incoming order demand rather than completed deliveries alone.

## 3. Exploratory Analysis and Decomposition

Hourly demand was visualized over time and compared by hour of day and day of week. Additive seasonal decomposition was performed using a 24-hour period to examine the observed series, trend, seasonal component, and residuals.

The observed series showed substantial hour-to-hour variation, including many zero-order hours and occasional demand spikes. The decomposition provided an exploratory view of daily seasonal structure and unexplained variation.

## 4. Stationarity Test

The Augmented Dickey-Fuller test was applied to the original hourly demand series.

- ADF statistic: -65.5493
- p-value: below 0.000001
- 5% critical value: -2.8622

The null hypothesis of a unit root was rejected at the 5% significance level. The series therefore appears stationary under this test. ARIMA(1,0,1), with no differencing, was used as a baseline specification. This order was not claimed to be globally optimal.

## 5. Forecasting Methodology

The observations were divided chronologically, using the first 80% for training and the final 20% for testing.

- Training observations: 3,474
- Testing observations: 869
- Training end: May 25, 2025, at 18:00
- Testing start: May 25, 2025, at 19:00

Three approaches were compared:

1. ARIMA(1,0,1)
2. Prophet with daily and weekly seasonality
3. A seasonal-naive baseline using the previous day's forecast for the corresponding hour, recursively extended through the test period

MAE and RMSE measured absolute forecast error. WAPE was used as an aggregate percentage-based metric. MAPE was calculated only on test hours with nonzero actual demand because zero actual values make ordinary MAPE undefined.

## 6. Model Comparison

| Model | MAE | RMSE | WAPE | MAPE on nonzero hours |
|---|---:|---:|---:|---:|
| ARIMA(1,0,1) | 0.8018 | 1.0224 | 70.8118% | 30.5959% |
| Prophet | 0.8056 | 1.0222 | 71.1480% | 31.1945% |
| Seasonal naive (24h) | 0.9965 | 1.3217 | 88.0081% | 55.0883% |

ARIMA achieved the lowest MAE and WAPE, while Prophet had a marginally lower RMSE. The differences between ARIMA and Prophet were small. Both statistical models outperformed the seasonal-naive baseline across the reported metrics.

ARIMA(1,0,1) was selected as the best baseline by MAE. Further model-order tuning and rolling-origin validation would be needed before treating it as the best production model.

## 7. Future Forecast

ARIMA and Prophet were refitted on all available observations to generate the next 24 hourly forecasts after the dataset's final timestamp. The seasonal-naive forecast was also generated for comparison.

The forecasts and chart were saved to the project outputs. The forecast period begins July 1, 2025, because the source dataset ends June 30, 2025.

## 8. Business Interpretation

The forecasts suggest that simple statistical models can provide a baseline estimate of hourly demand, but the high WAPE and visible demand spikes show that forecast accuracy remains limited.

In a real food-delivery marketplace, forecasts could inform staffing, driver availability, and restaurant preparation planning. These forecasts should not be used directly for operational decisions without validation on real data.

## 9. Limitations

- The dataset is synthetic and covers approximately six months.
- Hourly demand is sparse, with many zero-order hours.
- ARIMA(1,0,1) was used as a baseline rather than selected through comprehensive hyperparameter tuning.
- A single chronological holdout was used; rolling-origin validation was not performed.
- The 24-hour decomposition period was specified in advance and does not independently prove strong daily seasonality.
- The future forecast is a demonstration based on historical synthetic data, not a current operational forecast.

## 10. Deliverables

- `data/processed/hourly_demand.csv`
- `data/processed/hourly_demand_forecasts.csv`
- `data/processed/next_24_hour_demand_forecast.csv`
- `reports/adf_test_results.csv`
- `reports/forecast_model_comparison.csv`
- `reports/forecast_summary.txt`
- `reports/figures/hourly_demand_decomposition.png`
- `reports/figures/hourly_demand_forecast_comparison.png`
- `reports/figures/next_24_hour_demand_forecast.png`

## Conclusion

Day 5 established an hourly demand time series, examined decomposition and stationarity, compared ARIMA, Prophet, and a seasonal-naive baseline, and generated a 24-hour demonstration forecast. ARIMA(1,0,1) had the lowest MAE among the tested approaches, but the results show that demand variability remains difficult to predict accurately.
