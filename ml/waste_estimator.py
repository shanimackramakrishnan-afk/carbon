"""
waste_estimator.py
==================
Estimates bio / plastic / e-waste per household.
Applies fraud-confidence penalty so fraudulent data has reduced weight.
Produces 1-month, 3-month, and 1-year aggregated waste projections.

Reads  : output/final_household_carbon_scores.csv
         (output of carbon_points_model.py which already contains fraud scores)
Outputs: datasets/DATASET_WITH_TARGETS.csv
         reports/waste_estimates_summary.csv
         reports/waste_projections.csv
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("datasets", exist_ok=True)
os.makedirs("reports",  exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("output/final_household_carbon_scores.csv")
print(f"✔  Loaded {len(df):,} rows")

num_cols = ["milk_packets","deliveries","bottles","food_waste",
            "garden_waste","old_devices","batteries","fraud_probability"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# ─────────────────────────────────────────────
# 2. RAW WASTE ESTIMATION (kg-equivalent units)
# ─────────────────────────────────────────────
# Coefficients grounded in average household waste studies
df["bio_waste"] = (
    df["food_waste"]   * 0.65 +
    df["garden_waste"] * 0.42 +
    df["family_size_encoded"].fillna(2) * 0.18
)

df["plastic_waste"] = (
    df["milk_packets"] * 0.38 +
    df["bottles"]      * 0.44 +
    df["deliveries"]   * 0.22
)

df["e_waste"] = (
    df["old_devices"] * 0.72 +
    df["batteries"]   * 0.28
)

# ─────────────────────────────────────────────
# 3. FRAUD CONFIDENCE PENALTY
# Fraud households have unreliable self-reported data.
# We down-weight their waste contribution proportionally.
# SEVERE fraud → near-zero weight. MINOR → small reduction.
# ─────────────────────────────────────────────
severity_weight = {"SEVERE": 0.10, "MODERATE": 0.40, "MINOR": 0.70, "NONE": 1.00}
df["fraud_confidence_weight"] = df["fraud_severity"].map(severity_weight).fillna(1.0)

# Blend: (1 - fraud_prob/100) × severity_weight for double-layer dampening
df["penalty_weight"] = (
    (1 - df["fraud_probability"] / 100) * df["fraud_confidence_weight"]
).clip(lower=0.05)   # floor at 5% so record stays in dataset

df["bio_waste_adjusted"]     = (df["bio_waste"]     * df["penalty_weight"]).round(4)
df["plastic_waste_adjusted"] = (df["plastic_waste"] * df["penalty_weight"]).round(4)
df["e_waste_adjusted"]       = (df["e_waste"]       * df["penalty_weight"]).round(4)

print("✔  Fraud penalty applied")

# ─────────────────────────────────────────────
# 4. NORMALISE TARGETS  (0-1 for model training)
# ─────────────────────────────────────────────
target_cols = ["bio_waste_adjusted","plastic_waste_adjusted","e_waste_adjusted"]
scaler = MinMaxScaler()
df[target_cols] = scaler.fit_transform(df[target_cols]).round(6)
print("✔  Targets normalised")

# ─────────────────────────────────────────────
# 5. MULTI-HORIZON WASTE PROJECTIONS
# Uses raw (pre-normalised) waste values for realistic tonnage
# ─────────────────────────────────────────────
raw_bio     = df["bio_waste_adjusted"]     * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
raw_plastic = df["plastic_waste_adjusted"] * (scaler.data_max_[1] - scaler.data_min_[1]) + scaler.data_min_[1]
raw_ewaste  = df["e_waste_adjusted"]       * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]

total_bio     = raw_bio.sum()
total_plastic = raw_plastic.sum()
total_ewaste  = raw_ewaste.sum()

# Seasonal multipliers (bio-waste peaks in summer; plastic stable; e-waste spikes Q4)
seasonal_bio     = [1.0, 1.1, 1.2, 1.05, 0.95, 0.90, 1.0, 1.0, 0.95, 1.0, 1.1, 1.15]
seasonal_plastic = [1.0, 1.0, 1.0, 1.0,  1.05, 1.05, 1.1, 1.05, 1.0, 1.0, 1.1, 1.2 ]
seasonal_ewaste  = [0.9, 0.9, 1.0, 1.0,  1.0,  1.0,  0.9, 0.9, 1.0, 1.1, 1.2, 1.4 ]

months = pd.date_range(start="2026-03-01", periods=12, freq="MS")

projections = pd.DataFrame({
    "month":            months.strftime("%Y-%m"),
    "bio_waste_monthly":     [total_bio     * s for s in seasonal_bio],
    "plastic_waste_monthly": [total_plastic * s for s in seasonal_plastic],
    "e_waste_monthly":       [total_ewaste  * s for s in seasonal_ewaste],
})
projections["total_monthly"] = (
    projections["bio_waste_monthly"] +
    projections["plastic_waste_monthly"] +
    projections["e_waste_monthly"]
)
projections = projections.round(3)

# ─────────────────────────────────────────────
# 6. SUMMARY TABLE (1m / 3m / 12m)
# ─────────────────────────────────────────────
summary = {
    "Period":         ["1 Month (Mar 2026)","3 Months (Mar–May 2026)","12 Months (Full Year)"],
    "Bio Waste":      [
        projections["bio_waste_monthly"].iloc[0],
        projections["bio_waste_monthly"].iloc[:3].sum(),
        projections["bio_waste_monthly"].sum(),
    ],
    "Plastic Waste":  [
        projections["plastic_waste_monthly"].iloc[0],
        projections["plastic_waste_monthly"].iloc[:3].sum(),
        projections["plastic_waste_monthly"].sum(),
    ],
    "E-Waste":        [
        projections["e_waste_monthly"].iloc[0],
        projections["e_waste_monthly"].iloc[:3].sum(),
        projections["e_waste_monthly"].sum(),
    ],
    "Total":          [
        projections["total_monthly"].iloc[0],
        projections["total_monthly"].iloc[:3].sum(),
        projections["total_monthly"].sum(),
    ],
}
summary_df = pd.DataFrame(summary).round(3)

print("\n── Waste Projections ──")
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Household Waste Estimates", fontsize=14, fontweight="bold")

ax = axes[0]
ax.bar(["Bio","Plastic","E-Waste"],
       [projections["bio_waste_monthly"].iloc[0],
        projections["plastic_waste_monthly"].iloc[0],
        projections["e_waste_monthly"].iloc[0]],
       color=["#2ecc71","#3498db","#e74c3c"])
ax.set_title("1-Month Waste Estimate"); ax.set_ylabel("Units (adjusted)")

ax = axes[1]
ax.plot(projections["month"], projections["bio_waste_monthly"],     marker="o", label="Bio",     color="#2ecc71")
ax.plot(projections["month"], projections["plastic_waste_monthly"], marker="s", label="Plastic", color="#3498db")
ax.plot(projections["month"], projections["e_waste_monthly"],       marker="^", label="E-Waste", color="#e74c3c")
ax.set_title("12-Month Forecast"); ax.set_xlabel("Month")
ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reports/waste_estimates.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Plot saved → reports/waste_estimates.png")

# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
df.to_csv("datasets/DATASET_WITH_TARGETS.csv", index=False)
summary_df.to_csv("reports/waste_estimates_summary.csv", index=False)
projections.to_csv("reports/waste_projections.csv", index=False)

print("✔  Datasets saved → datasets/DATASET_WITH_TARGETS.csv")
print("✔  Reports saved  → reports/waste_estimates_summary.csv")
print("✔  Reports saved  → reports/waste_projections.csv")