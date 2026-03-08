import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ---------------- LOAD DATA ----------------
df = pd.read_csv("datasets/AI_cleaned_household_dataset.csv")
print("Initial shape:", df.shape)

# ---------------- MISSING VALUE DETECTION ----------------
print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------- CLEANING ----------------
# Remove duplicate rows based on user_email
df = df.drop_duplicates(subset=["user_email"])
print("\nAfter removing duplicates:", df.shape)

# Fill missing values
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values handled:")
print(df.isnull().sum())

# ---------------- ENCODING CATEGORICAL ----------------
categorical_cols = df.select_dtypes(include='object').columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
print("\nCategorical columns encoded")

# ---------------- NORMALIZATION ----------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Numeric columns normalized")

# ---------------- SAVE FINAL CLEAN DATASET ----------------
df.to_csv("FINAL_ML_READY_DATASET(3).csv", index=False)
print("\nFinal ML-ready dataset saved")
print("Final shape:", df.shape)
print(df.head())