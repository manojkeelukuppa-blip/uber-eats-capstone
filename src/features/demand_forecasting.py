
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hourly_demand.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

for directory in [FIGURES_DIR, REPORTS_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

TEST_FRACTION = 0.20
SEASONAL_PERIOD = 24

# --------------------------------------------------
# 2. Load hourly demand and chronological split
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")
df = df.asfreq("h")
df["order_count"] = df["order_count"].fillna(0)
series = df["order_count"].astype(float)

if len(series) < 100:
    raise ValueError("Not enough observations for forecasting.")

split_index = int(len(series) * (1 - TEST_FRACTION))
train = series.iloc[:split_index]
test = series.iloc[split_index:]

print("Chronological train/test split")
print("-" * 40)
print("Total observations:", len(series))
print("Training observations:", len(train))
print("Testing observations:", len(test))
print("Training ends:", train.index.max())
print("Testing begins:", test.index.min())

# --------------------------------------------------
# 3. Evaluation metrics
# --------------------------------------------------
def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    total_actual = actual.sum()
    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    nonzero_mask = actual > 0
    mape_nonzero = (
        np.mean(
            np.abs(
                (actual[nonzero_mask] - predicted[nonzero_mask])
                / actual[nonzero_mask]
            )
        ) * 100
        if nonzero_mask.any()
        else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
        "MAPE_nonzero_percent": mape_nonzero,
    }

# --------------------------------------------------
# 4. Seasonal-naive baseline
# --------------------------------------------------
# Forecast each test hour using the matching hour from
# the previous day. After the first test day, use the
# forecast from 24 hours earlier rather than test actuals.
seasonal_naive_values = list(train.iloc[-SEASONAL_PERIOD:])

for _ in range(len(test)):
    seasonal_naive_values.append(
        seasonal_naive_values[-SEASONAL_PERIOD]
    )

seasonal_naive_forecast = pd.Series(
    seasonal_naive_values[SEASONAL_PERIOD:],
    index=test.index,
    dtype=float,
).clip(lower=0)

# --------------------------------------------------
# 5. ARIMA baseline
# --------------------------------------------------
arima_order = (1, 0, 1)

print("\nFitting ARIMA", arima_order, "...")
arima_model = ARIMA(
    train,
    order=arima_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
arima_fit = arima_model.fit()

arima_forecast = pd.Series(
    np.asarray(arima_fit.forecast(steps=len(test))),
    index=test.index,
).clip(lower=0)

# --------------------------------------------------
# 6. Prophet baseline
# --------------------------------------------------
print("Fitting Prophet...")

prophet_train = train.rename("y").reset_index()
prophet_train.columns = ["ds", "y"]

prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    seasonality_mode="additive",
    uncertainty_samples=0,
)
prophet_model.fit(prophet_train)

prophet_output = prophet_model.predict(
    pd.DataFrame({"ds": test.index})
)

prophet_forecast = pd.Series(
    prophet_output["yhat"].to_numpy(),
    index=test.index,
).clip(lower=0)

# --------------------------------------------------
# 7. Compare models on the same test timestamps
# --------------------------------------------------
results = []

for model_name, predictions in [
    ("ARIMA(1,0,1)", arima_forecast),
    ("Prophet", prophet_forecast),
    ("Seasonal Naive (24h)", seasonal_naive_forecast),
]:
    metrics = calculate_metrics(test, predictions)
    metrics["Model"] = model_name
    results.append(metrics)

comparison = pd.DataFrame(results)[
    [
        "Model",
        "MAE",
        "RMSE",
        "WAPE_percent",
        "MAPE_nonzero_percent",
    ]
]

comparison_path = REPORTS_DIR / "forecast_model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print("\nModel comparison")
print("-" * 70)
print(comparison.round(4).to_string(index=False))

# --------------------------------------------------
# 8. Save actual and predicted values
# --------------------------------------------------
forecast_results = pd.DataFrame({
    "timestamp": test.index,
    "actual_orders": test.to_numpy(),
    "arima_forecast": arima_forecast.to_numpy(),
    "prophet_forecast": prophet_forecast.to_numpy(),
    "seasonal_naive_forecast": seasonal_naive_forecast.to_numpy(),
})

forecast_path = PROCESSED_DIR / "hourly_demand_forecasts.csv"
forecast_results.to_csv(forecast_path, index=False)

# --------------------------------------------------
# 9. Plot actual demand and forecasts
# --------------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(test.index, test.values, label="Actual demand", linewidth=1)
plt.plot(
    test.index, arima_forecast.values,
    label="ARIMA forecast", linewidth=1
)
plt.plot(
    test.index, prophet_forecast.values,
    label="Prophet forecast", linewidth=1
)
plt.plot(
    test.index, seasonal_naive_forecast.values,
    label="Seasonal-naive forecast", linewidth=1
)
plt.title("Hourly Order Demand: Actual vs Forecast")
plt.xlabel("Timestamp")
plt.ylabel("Orders per hour")
plt.legend()
plt.tight_layout()

forecast_plot_path = FIGURES_DIR / "hourly_demand_forecast_comparison.png"
plt.savefig(forecast_plot_path, dpi=150)
plt.close()

# --------------------------------------------------
# 10. Save summary
# --------------------------------------------------
best_model = comparison.loc[comparison["MAE"].idxmin(), "Model"]

summary_path = REPORTS_DIR / "forecast_summary.txt"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write("Hourly Demand Forecasting Summary\n")
    file.write("=" * 40 + "\n")
    file.write(f"ARIMA order: {arima_order}\n")
    file.write(f"Training observations: {len(train)}\n")
    file.write(f"Testing observations: {len(test)}\n")
    file.write(f"Training end: {train.index.max()}\n")
    file.write(f"Testing start: {test.index.min()}\n\n")
    file.write(comparison.to_string(index=False))
    file.write(f"\n\nBest model by MAE: {best_model}\n")
    file.write(
        "\nMAPE is calculated only for hours with nonzero actual demand; "
        "zero-demand hours are excluded.\n"
    )
    file.write(
        "WAPE uses all test hours and is reported as an additional metric.\n"
    )

print("\nBest model by MAE:", best_model)
print("Saved comparison:", comparison_path)
print("Saved forecast dataset:", forecast_path)
print("Saved plot:", forecast_plot_path)
print("Saved summary:", summary_path)

# --------------------------------------------------
# 11. Forecast the next 24 hours using all history
# --------------------------------------------------
HORIZON = 24

print("\nGenerating next 24-hour demand forecast...")

# Refit ARIMA on all available observations
final_arima_fit = ARIMA(
    series,
    order=arima_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit()

future_index = pd.date_range(
    start=series.index[-1] + pd.Timedelta(hours=1),
    periods=HORIZON,
    freq="h",
)

future_arima = pd.Series(
    np.asarray(final_arima_fit.forecast(steps=HORIZON)),
    index=future_index,
).clip(lower=0)

# Refit Prophet on all available observations
full_prophet_train = series.rename("y").reset_index()
full_prophet_train.columns = ["ds", "y"]

final_prophet_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    seasonality_mode="additive",
    uncertainty_samples=0,
)

final_prophet_model.fit(full_prophet_train)

future_dates = pd.DataFrame({"ds": future_index})
future_prophet_output = final_prophet_model.predict(future_dates)

future_prophet = pd.Series(
    future_prophet_output["yhat"].to_numpy(),
    index=future_index,
).clip(lower=0)

# Seasonal-naive forecast repeats the most recent 24-hour pattern
future_naive = pd.Series(
    series.iloc[-HORIZON:].to_numpy(),
    index=future_index,
)

future_results = pd.DataFrame({
    "timestamp": future_index,
    "arima_forecast": future_arima.to_numpy(),
    "prophet_forecast": future_prophet.to_numpy(),
    "seasonal_naive_forecast": future_naive.to_numpy(),
})

future_path = PROCESSED_DIR / "next_24_hour_demand_forecast.csv"
future_results.to_csv(future_path, index=False)

# Plot the future forecasts
plt.figure(figsize=(13, 5))
plt.plot(
    future_index, future_arima,
    label="ARIMA", marker="o", markersize=3
)
plt.plot(
    future_index, future_prophet,
    label="Prophet", marker="o", markersize=3
)
plt.plot(
    future_index, future_naive,
    label="Seasonal naive", marker="o", markersize=3
)
plt.title("Forecast for the Next 24 Hours")
plt.xlabel("Timestamp")
plt.ylabel("Forecast orders per hour")
plt.legend()
plt.tight_layout()

future_plot_path = FIGURES_DIR / "next_24_hour_demand_forecast.png"
plt.savefig(future_plot_path, dpi=150)
plt.close()

print("Forecast horizon:", HORIZON, "hours")
print("Forecast starts:", future_index.min())
print("Forecast ends:", future_index.max())
print("Saved future forecast:", future_path)
print("Saved future forecast plot:", future_plot_path)
print("\nFuture forecast preview:")
print(future_results.head(10).round(3).to_string(index=False))
