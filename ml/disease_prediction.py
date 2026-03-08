import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import joblib


# ---------------------------
# LOAD DATASET
# ---------------------------

df = pd.read_csv("FINAL_ML_READY_DATASET(3).csv")


# ---------------------------
# NUMERIC CONVERSION
# ---------------------------

numeric_cols = [
    "milk_packets",
    "bottles",
    "food_waste",
    "garden_waste",
    "old_devices",
    "batteries"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------
# ENCODE CATEGORICAL FEATURES
# ---------------------------

cat_cols = [
    "segregation",
    "compost",
    "disposal_method",
    "district"
]

label_encoders = {}

for col in cat_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = encoder


# ---------------------------
# HANDLE MISSING VALUES
# ---------------------------

print("\nMissing Values Before Fix:\n")
print(df.isnull().sum())

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())


# ---------------------------
# DISEASE RISK ENGINEERING
# ---------------------------

df["dengue_risk"] = (
    df["bottles"] * 3 +
    df["milk_packets"] * 1.5 +
    (df["segregation"] == 0) * 5
)

df["cholera_risk"] = (
    df["food_waste"] * 3 +
    (df["compost"] == 0) * 4
)

df["typhoid_risk"] = (
    df["food_waste"] * 2 +
    df["garden_waste"] * 1.5
)

df["respiratory_risk"] = (
    (df["disposal_method"] == 0) * 6 +
    df["milk_packets"] * 1.2
)

df["toxic_risk"] = (
    df["old_devices"] * 2.5 +
    df["batteries"] * 3
)


# ---------------------------
# GEOGRAPHIC RISK
# ---------------------------

district_risk = {
    0:4,
    1:3,
    2:5,
    3:6
}

df["geo_risk"] = df["district"].map(district_risk)
df["geo_risk"] = df["geo_risk"].fillna(4)


# ---------------------------
# TOTAL RISK SCORE
# ---------------------------

df["total_risk"] = (

    df["dengue_risk"] * 2 +
    df["cholera_risk"] * 2 +
    df["typhoid_risk"] * 1.5 +
    df["respiratory_risk"] * 1.5 +
    df["toxic_risk"] +
    df["geo_risk"]

)


# ---------------------------
# RISK LEVEL CLASSIFICATION
# ---------------------------

df["risk_level"] = pd.qcut(
    df["total_risk"],
    q=4,
    labels=["LOW","MODERATE","HIGH","CRITICAL"]
)

print("\nRisk Level Distribution:\n")
print(df["risk_level"].value_counts())


# ---------------------------
# CREATE DISEASE LABELS
# ---------------------------

df["dengue_label"] = (df["dengue_risk"] > df["dengue_risk"].median()).astype(int)

df["cholera_label"] = (df["cholera_risk"] > df["cholera_risk"].median()).astype(int)

df["typhoid_label"] = (df["typhoid_risk"] > df["typhoid_risk"].median()).astype(int)

df["respiratory_label"] = (df["respiratory_risk"] > df["respiratory_risk"].median()).astype(int)

df["toxic_label"] = (df["toxic_risk"] > df["toxic_risk"].median()).astype(int)


# ---------------------------
# FEATURES
# ---------------------------

features = [

    "milk_packets",
    "bottles",
    "food_waste",
    "garden_waste",
    "old_devices",
    "batteries",

    "segregation",
    "compost",
    "disposal_method",

    "dengue_risk",
    "cholera_risk",
    "typhoid_risk",
    "respiratory_risk",
    "toxic_risk",

    "geo_risk"

]

X = df[features]

y_risk = df["risk_level"]

y_disease = df[[

    "dengue_label",
    "cholera_label",
    "typhoid_label",
    "respiratory_label",
    "toxic_label"

]]


# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------

X_train, X_test, y_risk_train, y_risk_test = train_test_split(

    X,
    y_risk,
    test_size=0.2,
    random_state=42,
    stratify=y_risk

)

X_train2, X_test2, y_dis_train, y_dis_test = train_test_split(

    X,
    y_disease,
    test_size=0.2,
    random_state=42

)


# ---------------------------
# MODEL 1: RISK LEVEL MODEL
# ---------------------------

risk_model = RandomForestClassifier(

    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42

)

risk_model.fit(X_train, y_risk_train)

risk_pred = risk_model.predict(X_test)

print("\nRisk Level Accuracy:", accuracy_score(y_risk_test, risk_pred))


print("\nRisk Classification Report\n")

print(classification_report(y_risk_test, risk_pred))


print("\nConfusion Matrix\n")

print(confusion_matrix(y_risk_test, risk_pred))


# ---------------------------
# CROSS VALIDATION
# ---------------------------

cv_scores = cross_val_score(risk_model, X, y_risk, cv=5)

print("\nCross Validation Scores:", cv_scores)

print("Average CV Score:", cv_scores.mean())


# ---------------------------
# MODEL 2: MULTI DISEASE MODEL
# ---------------------------

base_model = RandomForestClassifier(

    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    random_state=42

)

multi_disease_model = MultiOutputClassifier(base_model)

multi_disease_model.fit(X_train2, y_dis_train)


# ---------------------------
# DISEASE PREDICTIONS
# ---------------------------

disease_pred = multi_disease_model.predict(X_test2)

diseases = [

    "Dengue",
    "Cholera",
    "Typhoid",
    "Respiratory",
    "Toxic Exposure"

]

for i, disease in enumerate(diseases):

    print("\n=====================")

    print(disease, "Prediction Report")

    print("=====================")

    print(classification_report(

        y_dis_test.iloc[:, i],
        disease_pred[:, i]

    ))


# ---------------------------
# FEATURE IMPORTANCE
# ---------------------------

importance = pd.DataFrame({

    "Feature": features,
    "Importance": risk_model.feature_importances_

}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")

print(importance)


# ---------------------------
# SAVE MODELS
# ---------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(risk_model, "models/risk_level_model.pkl")

joblib.dump(multi_disease_model, "models/multi_disease_model.pkl")

print("\nModels Saved Successfully")