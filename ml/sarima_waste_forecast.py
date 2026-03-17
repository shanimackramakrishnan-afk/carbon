"""
sarima_waste_forecast.py
=========================
SARIMA time-series forecasting for bio / plastic / e-waste.
Includes confidence intervals, model diagnostics, ADF stationarity test,
and 1-month / 3-month / 12-month forecast exports.

Reads  : datasets/DATASET_WITH_TARGETS.csv
Outputs: reports/future_waste_forecast.csv
         reports/sarima_forecast_bio.png
         reports/sarima_forecast_plastic.png
         reports/sarima_forecast_ewaste.png
         reports/sarima_diagnostics.txt
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & BUILD TIME SERIES
# ─────────────────────────────────────────────
df = pd.read_csv("datasets/DATASET_WITH_TARGETS.csv")
print(f"✔  Loaded {len(df):,} rows")

# Assign monthly dates — each row = one household recorded in sequence
# Aggregate by month to build a monthly waste time series
df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
df = df.set_index("date")

monthly = df[["bio_waste_adjusted","plastic_waste_adjusted","e_waste_adjusted"]].resample("ME").mean()

# Forward-fill any gaps
monthly = monthly.ffill()
print(f"✔  Monthly series: {len(monthly)} periods")

# ─────────────────────────────────────────────
# 2. STATIONARITY TEST (ADF)
# ─────────────────────────────────────────────
diag_lines = []
print("\n── ADF Stationarity Tests ──")
for col in monthly.columns:
    result = adfuller(monthly[col].dropna())
    stat, p = result[0], result[1]
    status = "Stationary ✔" if p < 0.05 else "Non-Stationary ✘"
    line = f"{col:<30}  ADF={stat:.4f}  p={p:.4f}  → {status}"
    print(f"   {line}")
    diag_lines.append(line)

# ─────────────────────────────────────────────
# 3. SARIMA PARAMETERS
# Seasonal period = 12 months. d=1 for differencing.
# If series is already stationary (ADF p<0.05), use d=0.
# ─────────────────────────────────────────────
def get_order(series):
    p_val = adfuller(series.dropna())[1]
    d = 0 if p_val < 0.05 else 1
    return (1, d, 1), (1, 1, 1, 12)

# ─────────────────────────────────────────────
# 4. FIT SARIMA & FORECAST
# ─────────────────────────────────────────────
FORECAST_STEPS = 12   # months ahead
results_store = {}

for col, label in zip(
    ["bio_waste_adjusted","plastic_waste_adjusted","e_waste_adjusted"],
    ["Bio Waste","Plastic Waste","E-Waste"]
):
    series = monthly[col].dropna()
    order, seasonal_order = get_order(series)

    print(f"\n── {label} — fitting SARIMA{order}×{seasonal_order} ──")
    sarima = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = sarima.fit(disp=False)
    print(f"   AIC={fitted.aic:.2f}  BIC={fitted.bic:.2f}")

    # Ljung-Box residual test (white noise check)
    lb = acorr_ljungbox(fitted.resid, lags=[10], return_df=True)
    lb_p = lb["lb_pvalue"].values[0]
    residual_status = "Residuals ~ white noise ✔" if lb_p > 0.05 else f"Autocorrelation in residuals (p={lb_p:.4f})"
    print(f"   Ljung-Box p={lb_p:.4f} → {residual_status}")
    diag_lines.append(f"{label}: AIC={fitted.aic:.2f}  BIC={fitted.bic:.2f}  {residual_status}")

    forecast_obj = fitted.get_forecast(steps=FORECAST_STEPS)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci   = forecast_obj.conf_int(alpha=0.05)

    results_store[col] = {
        "fitted":       fitted,
        "series":       series,
        "forecast":     forecast_mean,
        "ci_lower":     forecast_ci.iloc[:, 0],
        "ci_upper":     forecast_ci.iloc[:, 1],
        "label":        label,
    }

# ─────────────────────────────────────────────
# 5. BUILD FORECAST DATAFRAME
# ─────────────────────────────────────────────
future_index = pd.date_range(
    start=monthly.index[-1] + pd.DateOffset(months=1),
    periods=FORECAST_STEPS, freq="ME")

forecast_df = pd.DataFrame(index=future_index)
forecast_df.index.name = "month"

for col, res in results_store.items():
    forecast_df[f"{col}_forecast"]   = res["forecast"].values
    forecast_df[f"{col}_ci_lower"]   = res["ci_lower"].values
    forecast_df[f"{col}_ci_upper"]   = res["ci_upper"].values

forecast_df = forecast_df.round(6)

# ─────────────────────────────────────────────
# 6. PRINT HORIZON SUMMARIES
# ─────────────────────────────────────────────
print("\n── Forecast Summaries ──")
for label_key, col in [("Bio","bio_waste_adjusted"),
                        ("Plastic","plastic_waste_adjusted"),
                        ("E-Waste","e_waste_adjusted")]:
    fc = forecast_df[f"{col}_forecast"]
    print(f"\n   {label_key}")
    print(f"   1-month  : {fc.iloc[0]:.5f}")
    print(f"   3-month  : {fc.iloc[:3].mean():.5f} (avg)")
    print(f"   12-month : {fc.sum():.5f} (total)")

# ─────────────────────────────────────────────
# 7. PLOTS (one per waste type)
# ─────────────────────────────────────────────
colors = {"bio_waste_adjusted":"#2ecc71",
          "plastic_waste_adjusted":"#3498db",
          "e_waste_adjusted":"#e74c3c"}

for col, res in results_store.items():
    fig, ax = plt.subplots(figsize=(11, 5))

    # Historical
    ax.plot(res["series"].index, res["series"].values,
            color="#555", lw=1.5, label="Historical")

    # Fitted values
    ax.plot(res["fitted"].fittedvalues.index,
            res["fitted"].fittedvalues.values,
            color="#aaa", lw=1, linestyle="--", label="Fitted")

    # Forecast
    fc_idx = forecast_df.index
    ax.plot(fc_idx, res["forecast"].values,
            color=colors[col], lw=2, marker="o", markersize=4, label="Forecast")

    # Confidence interval
    ax.fill_between(fc_idx,
                    forecast_df[f"{col}_ci_lower"],
                    forecast_df[f"{col}_ci_upper"],
                    color=colors[col], alpha=0.15, label="95% CI")

    ax.axvline(monthly.index[-1], color="black", lw=1, linestyle=":")
    ax.set_title(f"{res['label']} — SARIMA Forecast (12 Months)", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Normalised Waste")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fname = f"reports/sarima_forecast_{col.split('_')[0]}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔  Plot saved → {fname}")

# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
forecast_df.to_csv("reports/future_waste_forecast.csv")

clean_diag = [line.replace("✔", "[OK]").replace("✘", "[FAIL]") for line in diag_lines]
with open("reports/sarima_diagnostics.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(clean_diag))

print("\n✔  Forecast saved → reports/future_waste_forecast.csv")
print("✔  Diagnostics  → reports/sarima_diagnostics.txt")
print(f"\nFull 12-month forecast:\n{forecast_df[['bio_waste_adjusted_forecast','plastic_waste_adjusted_forecast','e_waste_adjusted_forecast']].to_string()}")