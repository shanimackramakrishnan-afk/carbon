"""
train_waste_prediction_model.py
================================
Multi-output RandomForest regression to predict
bio / plastic / e-waste per household.

Reads  : datasets/DATASET_WITH_TARGETS.csv
Outputs: models/waste_prediction_model.pkl
         models/waste_scaler.pkl
         reports/waste_model_evaluation.csv
         reports/waste_feature_importance.png
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("datasets/DATASET_WITH_TARGETS.csv")
print(f"✔  Loaded {len(df):,} rows")

# ─────────────────────────────────────────────
# 2. FEATURES & TARGETS
# ─────────────────────────────────────────────
feature_cols = [
    "family_size_encoded","milk_packets","deliveries","bottles",
    "food_waste","garden_waste","old_devices","batteries",
    "fraud_probability","fraud_confidence_weight",
    "carbon_points","carbon_trust_score",
    "milk_per_person","deliveries_per_person","devices_per_person",
    "waste_per_person","milk_x_delivery",
]

# Keep only features that exist in the loaded dataframe
feature_cols = [c for c in feature_cols if c in df.columns]

target_cols = ["bio_waste_adjusted","plastic_waste_adjusted","e_waste_adjusted"]

# Drop rows with NaN in features or targets
df_model = df[feature_cols + target_cols].dropna()
print(f"✔  Model rows after dropna: {len(df_model):,}")

X = df_model[feature_cols].values
y = df_model[target_cols].values

# ─────────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────────
feat_scaler = RobustScaler()
X_scaled = feat_scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42)

# ─────────────────────────────────────────────
# 5. MODEL — Tuned RandomForest
# ─────────────────────────────────────────────
base_rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
model = MultiOutputRegressor(base_rf, n_jobs=-1)
model.fit(X_train, y_train)
print("✔  Model trained")

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

eval_rows = []
target_names = ["Bio Waste","Plastic Waste","E-Waste"]
for i, name in enumerate(target_names):
    mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2   = r2_score(y_test[:, i], y_pred[:, i])
    eval_rows.append({"Target": name, "MAE": round(mae,5),
                      "RMSE": round(rmse,5), "R2": round(r2,4)})
    print(f"   {name:<15}  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.4f}")

eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv("reports/waste_model_evaluation.csv", index=False)

# Overall R² (multioutput)
overall_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
print(f"\n✔  Overall R²: {overall_r2:.4f}")

# ─────────────────────────────────────────────
# 7. CROSS-VALIDATION (per target)
# ─────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("\n── 5-Fold CV R² per target ──")
for i, name in enumerate(target_names):
    single = RandomForestRegressor(n_estimators=100, max_depth=12,
                                   max_features="sqrt", random_state=42, n_jobs=-1)
    cv = cross_val_score(single, X_scaled, y[:, i], cv=kf, scoring="r2")
    print(f"   {name:<15}  mean R²={cv.mean():.4f}  std={cv.std():.4f}")

# ─────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}
                      ).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(5, len(feature_cols)*0.35)))
ax.barh(imp_df["Feature"], imp_df["Importance"], color="#378add")
ax.set_title("Waste Behaviour Impact on Waste Generation", fontsize=13)
ax.set_xlabel("Mean Feature Importance")
plt.tight_layout()
plt.savefig("reports/waste_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Plot saved → reports/waste_feature_importance.png")

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
joblib.dump(model,       "models/waste_prediction_model.pkl")
joblib.dump(feat_scaler, "models/waste_feature_scaler.pkl")
print("✔  Models saved → models/")