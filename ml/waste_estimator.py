import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("FINAL_ML_READY_DATASET(3).csv")

# -----------------------------
# CREATE TARGET WASTE COLUMNS
# -----------------------------

df["bio_waste"] = (
    df["food_waste"] * 0.6 +
    df["garden_waste"] * 0.4 +
    df["family_size"] * 0.2
)

df["plastic_waste"] = (
    df["milk_packets"] * 0.4 +
    df["bottles"] * 0.4 +
    df["deliveries"] * 0.2
)

df["e_waste"] = (
    df["old_devices"] * 0.7 +
    df["batteries"] * 0.3
)

print("Waste columns created")

# -----------------------------
# CREATE FRAUD PENALTY WEIGHT
# -----------------------------

df["fraud_penalty_weight"] = 1 - df["fraud_probability"]

print("Fraud penalty weight created")

# -----------------------------
# APPLY PENALTY TO WASTE
# -----------------------------

df["bio_waste_adjusted"] = df["bio_waste"] * df["fraud_penalty_weight"]
df["plastic_waste_adjusted"] = df["plastic_waste"] * df["fraud_penalty_weight"]
df["e_waste_adjusted"] = df["e_waste"] * df["fraud_penalty_weight"]

print("Fraud penalty applied to waste")

# -----------------------------
# NORMALIZE FINAL TARGETS
# -----------------------------

scaler = MinMaxScaler()

target_cols = [
    "bio_waste_adjusted",
    "plastic_waste_adjusted",
    "e_waste_adjusted"
]

df[target_cols] = scaler.fit_transform(df[target_cols])

print("Adjusted waste normalized")

# -----------------------------
# SAVE DATASET
# -----------------------------

df.to_csv("datasets/DATASET_WITH_TARGETS.csv", index=False)

print("Final dataset saved successfully")