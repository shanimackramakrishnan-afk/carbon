import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from waste_estimator import estimate_waste   # IMPORT ESTIMATOR

print("Loading Dataset...")

df = pd.read_csv("datasets/FINAL_ML_READY_DATASET.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ---------------- CREATE WASTE COLUMNS ----------------

print("Estimating Monthly Waste...")

df[
    [
        "plastic_waste_kg",
        "bio_waste_kg",
        "ewaste_kg",
        "monthly_waste_kg"
    ]
] = df.apply(estimate_waste, axis=1)

print("Waste columns created")

# ---------------- ENCODE CATEGORICAL COLUMNS ----------------

categorical_columns = [
    "house_type",
    "district",
    "oil_type",
    "segregation",
    "food_waste",
    "compost",
    "garden_waste",
    "disposal_method"
]

label_encoders = {}

for col in categorical_columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder

print("Categorical Encoding Completed")

# ---------------- DEFINE FEATURES ----------------
# IMPORTANT: Remove derived waste columns

X = df.drop(columns=[
    "user_email",
    "plastic_waste_kg",
    "bio_waste_kg",
    "ewaste_kg",
    "monthly_waste_kg"
])

y = df["monthly_waste_kg"]

print("Training Features:")
print(X.columns)

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Size:", len(X_train))
print("Testing Size:", len(X_test))

# ---------------- TRAIN MODEL ----------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

# ---------------- PREDICTION ----------------

y_pred = model.predict(X_test)

# ---------------- PERFORMANCE ----------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE:", mae)
print("R2 Score:", r2)

# ---------------- FEATURE IMPORTANCE ----------------

importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance Ranking")
print(importance_df)

# ---------------- FEATURE IMPORTANCE GRAPH ----------------

plt.figure(figsize=(10,6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("AI Feature Importance for Waste Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# ---------------- SAVE MODEL ----------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/waste_prediction_model.pkl")

print("\nModel saved successfully")

# ---------------- SAVE ENCODERS ----------------

joblib.dump(label_encoders, "models/label_encoders.pkl")

print("Encoders saved successfully")