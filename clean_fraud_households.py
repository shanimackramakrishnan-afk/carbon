import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("Loading Fraud Household Dataset...")

# Load dataset
df = pd.read_csv("datasets/fraud_households_detected.csv")

print("Original Shape:", df.shape)

# ---------------------------
# 1️⃣ Remove duplicate rows
# ---------------------------
df = df.drop_duplicates()

# ---------------------------
# 2️⃣ Standardize column names
# ---------------------------
df.columns = df.columns.str.strip().str.lower()

# ---------------------------
# 3️⃣ Define columns
# ---------------------------
numeric_cols = [
    "family_size",
    "milk_packets",
    "deliveries",
    "bottles",
    "food_waste",
    "garden_waste",
    "old_devices",
    "batteries"
]

text_cols = [
    "house_type",
    "district",
    "oil_type",
    "segregation",
    "compost",
    "disposal_method"
]

# ---------------------------
# 4️⃣ Convert numeric columns safely
# ---------------------------
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# 5️⃣ Handle missing values
# ---------------------------
for col in df.columns:

    if col in numeric_cols:

        # If entire column is NaN
        if df[col].isnull().all():
            df[col] = 0

        else:
            df[col] = df[col].fillna(df[col].median())

    else:
        df[col] = df[col].fillna("unknown")

# ---------------------------
# 6️⃣ Normalize text columns
# ---------------------------
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

# ---------------------------
# 7️⃣ Convert YES/NO columns
# ---------------------------
binary_cols = ["segregation", "compost"]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].replace({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "unknown": 0
        })

# ---------------------------
# 8️⃣ Encode categorical features
# ---------------------------
categorical_cols = [
    "house_type",
    "district",
    "oil_type",
    "disposal_method"
]

le = LabelEncoder()

for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# ---------------------------
# 9️⃣ Handle constant columns before scaling
# ---------------------------
for col in numeric_cols:
    if df[col].nunique() <= 1:
        df[col] = 0

# ---------------------------
# 🔟 Normalize numeric features
# ---------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---------------------------
# 1️⃣1️⃣ Remove email column
# ---------------------------
if "user_email" in df.columns:
    df = df.drop(columns=["user_email"])

# ---------------------------
# 1️⃣2️⃣ Save cleaned dataset
# ---------------------------
df.to_csv("fraud_household_cleaned.csv", index=False)

print("Cleaned Dataset Saved Successfully")
print("Final Shape:", df.shape)

# ---------------------------
# 1️⃣3️⃣ Verify no NaN values
# ---------------------------
print("\nRemaining Missing Values:")
print(df.isnull().sum())