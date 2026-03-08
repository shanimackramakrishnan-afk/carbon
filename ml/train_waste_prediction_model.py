import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv("datasets/DATASET_WITH_TARGETS.csv")

# -----------------------------
# DEFINE FEATURES
# -----------------------------

feature_cols = [
    "family_size",
    "house_type",
    "milk_packets",
    "deliveries",
    "bottles",
    "food_waste",
    "garden_waste",
    "old_devices",
    "batteries",
    "segregation",
    "compost",
    "disposal_method",
    "family_size_encoded",
    "batteries_encoded"
]

X = df[feature_cols]

# -----------------------------
# DEFINE TARGETS
# -----------------------------

target_cols = [
    "bio_waste_adjusted",
    "plastic_waste_adjusted",
    "e_waste_adjusted"
]

y = df[target_cols]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# CREATE MODEL
# -----------------------------

base_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42
)

model = MultiOutputRegressor(base_model)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model.fit(X_train, y_train)

print("Model training completed")

# -----------------------------
# PREDICT
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# EVALUATE MODEL
# -----------------------------

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(model, "waste_prediction_model.pkl")

print("Model saved successfully")

# ---------------------------------------------------
# WASTE BEHAVIOUR IMPACT GRAPH
# ---------------------------------------------------

# Collect feature importance from each waste model
importances = []

for estimator in model.estimators_:
    importances.append(estimator.feature_importances_)

# Average importance across all outputs
mean_importance = np.mean(importances, axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Impact": mean_importance
})

importance_df = importance_df.sort_values(by="Impact", ascending=True)

# -----------------------------
# PLOT BEHAVIOUR IMPACT GRAPH
# -----------------------------

plt.figure(figsize=(10,6))
plt.barh(importance_df["Feature"], importance_df["Impact"])

plt.title("Waste Behaviour Impact on Waste Generation")
plt.xlabel("Impact Score")
plt.ylabel("Household Behaviour")

plt.tight_layout()
plt.show()