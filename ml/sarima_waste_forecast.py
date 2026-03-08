import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("datasets/DATASET_WITH_TARGETS.csv")

# -----------------------------
# CREATE DATE COLUMN
# -----------------------------

df["date"] = pd.date_range(start="2026-03-01", periods=len(df), freq="ME")
df = df.set_index("date")

# -----------------------------
# SELECT WASTE TYPES
# -----------------------------

bio_series = df["bio_waste_adjusted"]
plastic_series = df["plastic_waste_adjusted"]
ewaste_series = df["e_waste_adjusted"]

# -----------------------------
# SARIMA PARAMETERS
# -----------------------------

order = (1,1,1)
seasonal_order = (1,1,1,12)

# -----------------------------
# TRAIN MODELS
# -----------------------------

bio_model = SARIMAX(bio_series, order=order, seasonal_order=seasonal_order)
plastic_model = SARIMAX(plastic_series, order=order, seasonal_order=seasonal_order)
ewaste_model = SARIMAX(ewaste_series, order=order, seasonal_order=seasonal_order)

bio_results = bio_model.fit()
plastic_results = plastic_model.fit()
ewaste_results = ewaste_model.fit()

print("SARIMA models trained successfully")

# -----------------------------
# FORECAST FUTURE (12 MONTHS)
# -----------------------------

forecast_steps = 12

bio_forecast = bio_results.forecast(steps=forecast_steps)
plastic_forecast = plastic_results.forecast(steps=forecast_steps)
ewaste_forecast = ewaste_results.forecast(steps=forecast_steps)

# -----------------------------
# CREATE FORECAST DATAFRAME
# -----------------------------

forecast_df = pd.DataFrame({
    "bio_waste_forecast": bio_forecast,
    "plastic_waste_forecast": plastic_forecast,
    "e_waste_forecast": ewaste_forecast
})

print("\nFuture Waste Forecast")
print(forecast_df)

# -----------------------------
# SAVE FORECAST FILE
# -----------------------------

forecast_df.to_csv("future_waste_forecast.csv")

print("\nForecast saved as future_waste_forecast.csv")

# -----------------------------
# PLOT RESULTS
# -----------------------------

# -----------------------------
# BIO WASTE FORECAST GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(bio_forecast, marker="o")
plt.title("Bio Waste Forecast (Next 12 Months)")
plt.xlabel("Month")
plt.ylabel("Predicted Bio Waste")
plt.grid(True)

plt.show()


# -----------------------------
# PLASTIC WASTE FORECAST GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(plastic_forecast, marker="o")
plt.title("Plastic Waste Forecast (Next 12 Months)")
plt.xlabel("Month")
plt.ylabel("Predicted Plastic Waste")
plt.grid(True)

plt.show()


# -----------------------------
# E-WASTE FORECAST GRAPH
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(ewaste_forecast, marker="o")
plt.title("E-Waste Forecast (Next 12 Months)")
plt.xlabel("Month")
plt.ylabel("Predicted E-Waste")
plt.grid(True)

plt.show()