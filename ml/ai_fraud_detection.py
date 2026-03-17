"""
ai_fraud_detection.py
=====================
Ensemble anomaly detection (IsolationForest + LOF + EllipticEnvelope).
Outputs:
  fraud_households_detected.csv
  datasets/AI_cleaned_household_dataset.csv
  datasets/fraud_scored_full.csv   <-- used by ALL downstream scripts
  models/fraud_detection_model.pkl
  models/robust_scaler.pkl
  models/family_encoder.pkl
  models/battery_encoder.pkl

Run this FIRST before any other script.
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")
os.makedirs("models",   exist_ok=True)
os.makedirs("datasets", exist_ok=True)

RANDOM_STATE = 42
FRAUD_THRESH = 70
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
df = pd.read_csv("FINAL_ML_READY_DATASET(2).csv")
print(f"✔  Loaded {len(df):,} rows | {df.shape[1]} columns")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"✔  Dropped {before - len(df)} duplicates")

num_cols = ["milk_packets","deliveries","old_devices","bottles","food_waste","garden_waste"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())

# ─────────────────────────────────────────────
# 2. ENCODE
# ─────────────────────────────────────────────
family_encoder  = LabelEncoder()
battery_encoder = LabelEncoder()
df["family_size_encoded"] = family_encoder.fit_transform(df["family_size"].astype(str))
df["batteries_encoded"]   = battery_encoder.fit_transform(df["batteries"].astype(str))

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
eps = 1e-6
df["milk_per_person"]       = df["milk_packets"]  / (df["family_size_encoded"] + eps)
df["deliveries_per_person"] = df["deliveries"]    / (df["family_size_encoded"] + eps)
df["devices_per_person"]    = df["old_devices"]   / (df["family_size_encoded"] + eps)
df["milk_x_delivery"]       = df["milk_packets"]  * df["deliveries"]
df["waste_per_person"]      = (df["food_waste"] + df["garden_waste"]) / (df["family_size_encoded"] + eps)

for col in ["milk_packets","deliveries","old_devices","food_waste"]:
    df[f"{col}_zscore"] = np.abs(stats.zscore(df[col].fillna(0)))

feature_cols = [
    "family_size_encoded","milk_packets","deliveries","old_devices",
    "batteries_encoded","milk_per_person","deliveries_per_person",
    "devices_per_person","milk_x_delivery","waste_per_person",
    "milk_packets_zscore","deliveries_zscore","old_devices_zscore","food_waste_zscore",
]

X_raw = df[feature_cols].copy()
scaler = RobustScaler()
X = scaler.fit_transform(X_raw)

# ─────────────────────────────────────────────
# 4. DYNAMIC CONTAMINATION
# ─────────────────────────────────────────────
def iqr_frac(arr):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return np.mean((arr < q1 - 1.5*iqr) | (arr > q3 + 1.5*iqr))

contamination = float(np.clip(
    np.mean([iqr_frac(X_raw[c].values) for c in feature_cols]), 0.01, 0.20))
print(f"✔  Auto contamination: {contamination:.3f}")

# ─────────────────────────────────────────────
# 5. ENSEMBLE ANOMALY DETECTION
# ─────────────────────────────────────────────
iso = IsolationForest(contamination=contamination, n_estimators=200, random_state=RANDOM_STATE)
iso.fit(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=False)
lof.fit_predict(X)

ee = EllipticEnvelope(contamination=contamination, random_state=RANDOM_STATE)
ee.fit(X)

def norm01(arr):
    r = arr.max() - arr.min()
    return (arr - arr.min()) / (r if r else 1)

ensemble = (
    (1 - norm01(iso.decision_function(X))) +
    norm01(-lof.negative_outlier_factor_) +
    (1 - norm01(ee.decision_function(X)))
) / 3

df["fraud_probability"] = (ensemble * 100).round(2)
df["fraud_flag"]        = (df["fraud_probability"] > FRAUD_THRESH).astype(int)

# ─────────────────────────────────────────────
# 6. FRAUD SEVERITY TIERS
# ─────────────────────────────────────────────
def severity(row):
    if row["fraud_flag"] == 0: return "NONE"
    p = row["fraud_probability"]
    return "SEVERE" if p > 90 else "MODERATE" if p > 80 else "MINOR"

df["fraud_severity"] = df.apply(severity, axis=1)

fraud_df = df[df["fraud_flag"] == 1].copy()
clean_df = df[df["fraud_flag"] == 0].copy()

print(f"\n⚠  Fraud detected : {len(fraud_df):,}  ({len(fraud_df)/len(df)*100:.1f}%)")
print(f"   MINOR    : {(df['fraud_severity']=='MINOR').sum()}")
print(f"   MODERATE : {(df['fraud_severity']=='MODERATE').sum()}")
print(f"   SEVERE   : {(df['fraud_severity']=='SEVERE').sum()}")

# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
fraud_df.to_csv("fraud_households_detected.csv", index=False)
clean_df.to_csv("datasets/AI_cleaned_household_dataset.csv", index=False)
df.to_csv("datasets/fraud_scored_full.csv", index=False)

joblib.dump(iso,             "models/fraud_detection_model.pkl")
joblib.dump(scaler,          "models/robust_scaler.pkl")
joblib.dump(family_encoder,  "models/family_encoder.pkl")
joblib.dump(battery_encoder, "models/battery_encoder.pkl")

print("\n✔  Models & datasets saved")
print("\nFraud Sample (top 10):")
print(fraud_df[["user_email","family_size","milk_packets","deliveries",
                "old_devices","batteries","fraud_probability","fraud_severity"]
               ].head(10).to_string(index=False))