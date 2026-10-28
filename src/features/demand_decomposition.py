
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "hourly_demand.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load hourly demand data
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Ensure a continuous hourly time series
df = df.asfreq("h")
df["order_count"] = df["order_count"].fillna(0)

series = df["order_count"].astype(float)

print("Hourly demand time-series analysis")
print("-" * 40)
print("Observations:", len(series))
print("Missing values:", series.isna().sum())
print("Date range:", series.index.min(), "to", series.index.max())
print("Mean hourly demand:", round(series.mean(), 3))
print("Hourly demand standard deviation:", round(series.std(), 3))

# Seasonal decomposition using a 24-hour daily period
decomposition = seasonal_decompose(
    series,
    model="additive",
    period=24,
    extrapolate_trend="freq",
)

# Plot observed, trend, seasonal, and residual components
fig = decomposition.plot()
fig.set_size_inches(14, 10)
fig.suptitle("Hourly Demand: Daily Seasonal Decomposition", y=1.02)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "hourly_demand_decomposition.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)

# Augmented Dickey-Fuller test on the original series
adf_result = adfuller(series, autolag="AIC")

adf_statistic = adf_result[0]
p_value = adf_result[1]
used_lags = adf_result[2]
observations = adf_result[3]
critical_values = adf_result[4]

print("\nADF test on original hourly demand")
print("-" * 40)
print(f"ADF statistic: {adf_statistic:.6f}")
print(f"p-value: {p_value:.6f}")
print("Lags used:", used_lags)
print("Observations used:", observations)

for level, value in critical_values.items():
    print(f"Critical value ({level}): {value:.6f}")

if p_value < 0.05:
    conclusion = (
        "Reject the unit-root null hypothesis at the 5% level. "
        "The series appears stationary under this test."
    )
else:
    conclusion = (
        "Fail to reject the unit-root null hypothesis at the 5% level. "
        "The series may be non-stationary; consider differencing "
        "and further diagnostics."
    )

print("\nInterpretation:", conclusion)

# Save ADF results for documentation
results = {
    "test": "Augmented Dickey-Fuller",
    "series": "original hourly order_count",
    "adf_statistic": adf_statistic,
    "p_value": p_value,
    "used_lags": used_lags,
    "observations_used": observations,
    "conclusion_at_5_percent": conclusion,
}

for level, value in critical_values.items():
    results[f"critical_value_{level}"] = value

results_path = REPORTS_DIR / "adf_test_results.csv"
pd.DataFrame([results]).to_csv(results_path, index=False)

print("\nSaved decomposition plot:",
      FIGURES_DIR / "hourly_demand_decomposition.png")
print("Saved ADF results:", results_path)
