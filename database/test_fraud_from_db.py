import sqlite3
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------

model = joblib.load("models/fraud_detection_model.pkl")
family_encoder = joblib.load("models/family_encoder.pkl")
battery_encoder = joblib.load("models/battery_encoder.pkl")

print("Model and encoders loaded")


# ---------------- CONNECT DATABASE ----------------

conn = sqlite3.connect("database/user.db")


# ---------------- JOIN ALL TABLES ----------------

query = """
SELECT 
    p.user_email,
    p.family_size,
    p.milk_packets,
    p.deliveries,
    e.old_devices,
    e.batteries,
    b.food_waste,
    b.compost,
    b.garden_waste
FROM plastic_data p
JOIN ewaste_data e
    ON p.user_email = e.user_email
JOIN biodegradable_data b
    ON p.user_email = b.user_email
"""

df = pd.read_sql_query(query, conn)

print("\nCombined ML Data:\n")
print(df)


# ---------------- ENCODE CATEGORICAL DATA ----------------

df["family_size_encoded"] = family_encoder.transform(df["family_size"].astype(str))
df["batteries_encoded"] = battery_encoder.transform(df["batteries"].astype(str))


# ---------------- SELECT FEATURES FOR MODEL ----------------

features = df[
[
    "family_size_encoded",
    "milk_packets",
    "deliveries",
    "old_devices",
    "batteries_encoded"
]
]


# ---------------- FRAUD DETECTION ----------------

predictions = model.predict(features)

df["fraud_prediction"] = predictions


# ---------------- SHOW RESULT ----------------

print("\nFraud Detection Result:\n")

print(df[
[
    "user_email",
    "family_size",
    "milk_packets",
    "deliveries",
    "old_devices",
    "batteries",
    "food_waste",
    "compost",
    "garden_waste",
    "fraud_prediction"
]])

print("\nPrediction meaning:")
print("1 = Normal")
print("-1 = Fraud")

conn.close()