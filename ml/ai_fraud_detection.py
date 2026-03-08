import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

# ---------------- LOAD DATA ----------------

df = pd.read_csv("datasets/2000_household_dataset_cleaned.csv")

print("Total rows in dataset:", len(df))


# ---------------- ENCODE CATEGORICAL ----------------

family_encoder = LabelEncoder()
battery_encoder = LabelEncoder()

df["family_size_encoded"] = family_encoder.fit_transform(df["family_size"])
df["batteries_encoded"] = battery_encoder.fit_transform(df["batteries"])


# ---------------- SELECT FEATURES ----------------

features = df[[
    "family_size_encoded",
    "milk_packets",
    "deliveries",
    "old_devices",
    "batteries_encoded"
]]


# ---------------- TRAIN MODEL ----------------

model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)

model.fit(features)

print("Fraud Detection Model Trained")


# ---------------- ANOMALY SCORE ----------------

scores = model.decision_function(features)

# convert to probability 0-100
fraud_probability = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100

df["fraud_probability"] = fraud_probability.round(2)


# ---------------- CLASSIFY FRAUD ----------------

df["fraud_flag"] = df["fraud_probability"].apply(
    lambda x: 1 if x > 70 else 0
)


# ---------------- SHOW FRAUD ROWS ----------------

fraud_rows = df[df["fraud_flag"] == 1]

print("\n⚠ Fraudulent households detected:", len(fraud_rows))

print("\nFraud Details:\n")

print(fraud_rows[[
    "user_email",
    "family_size",
    "milk_packets",
    "deliveries",
    "old_devices",
    "batteries",
    "fraud_probability"
]])


# ---------------- SAVE FRAUD DATA ----------------

fraud_rows.to_csv("fraud_households_detected.csv", index=False)

print("\nFraud rows saved to fraud_households_detected.csv")


# ---------------- CLEAN DATASET ----------------

clean_df = df[df["fraud_flag"] == 0]

clean_df.to_csv("datasets/AI_cleaned_household_dataset.csv", index=False)

print("Clean dataset saved as AI_cleaned_household_dataset.csv")


# ---------------- SAVE MODEL ----------------

joblib.dump(model, "models/fraud_detection_model.pkl")
print("Model saved")


# ---------------- SAVE ENCODERS ----------------

joblib.dump(family_encoder, "models/family_encoder.pkl")
joblib.dump(battery_encoder, "models/battery_encoder.pkl")

print("Encoders Saved")
