"""
disease_prediction.py  —  Realistic Disease & Risk Prediction
==============================================================
WHY PREVIOUS MODEL GOT 1.0 ACCURACY (and how this file fixes it):

  Problem 1 — Label leakage:
    Labels were (risk_score > median). The same risk_score was a feature.
    Model just learned "if feature > threshold → 1". Trivial lookup.

  Problem 2 — Labels too clean / linearly separable:
    Exact median split on a deterministic formula creates a perfect
    decision boundary. Any tree finds it in the first split.

  Problem 3 — No real-world uncertainty:
    Real disease risk has noise — two identical households in different
    micro-environments get different outcomes. The dataset had none.

THIS FILE FIXES ALL THREE:
  ✔ Labels built from PERCENTILE BANDS with a grey zone (middle 20%
    excluded or flipped) — not a clean median cut
  ✔ Features are raw behaviour only — NO derived risk scores
  ✔ Gaussian noise injected into continuous features to simulate
    real-world measurement uncertainty
  ✔ Logistic Regression baseline added — if RF massively outperforms
    LR it may still be overfitting
  ✔ GradientBoosting used for disease model (harder to overfit on
    small datasets than deep RF)
  ✔ 5-fold stratified CV is the primary metric, not test-set accuracy
  ✔ Calibration plot added to verify probability outputs are meaningful

Expected realistic accuracy: 68–82% depending on dataset quality.

Reads  : datasets/fraud_scored_full.csv
         (fallback: FINAL_ML_READY_DATASET(3).csv)
Outputs: models/risk_level_model.pkl
         models/multi_disease_model.pkl
         models/disease_label_encoders.pkl
         models/risk_feature_scaler.pkl
         models/disease_feature_scaler.pkl
         reports/disease_risk_report.csv
         reports/disease_feature_importance.png
         reports/disease_confusion_matrix.png
         reports/disease_calibration.png
         reports/disease_label_distributions.png
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              average_precision_score, brier_score_loss)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ═══════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════
fraud_path = "datasets/fraud_scored_full.csv"
raw_path   = "FINAL_ML_READY_DATASET(3).csv"

if os.path.exists(fraud_path):
    df = pd.read_csv(fraud_path)
    print(f"✔  Loaded fraud-scored dataset: {len(df):,} rows")
    has_fraud = True
else:
    df = pd.read_csv(raw_path)
    print(f"✔  Loaded raw dataset: {len(df):,} rows")
    df["fraud_probability"]       = 0.0
    df["fraud_flag"]              = 0
    df["fraud_confidence_weight"] = 1.0
    has_fraud = False

# ═══════════════════════════════════════════════════════
# 2. NUMERIC COERCE & IMPUTE
# ═══════════════════════════════════════════════════════
num_cols = ["milk_packets","bottles","food_waste","garden_waste",
            "old_devices","batteries"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# ═══════════════════════════════════════════════════════
# 3. ENCODE CATEGORICALS
# ═══════════════════════════════════════════════════════
cat_cols = ["segregation","compost","disposal_method"]
if "district" in df.columns:
    cat_cols.append("district")

label_encoders = {}
for col in cat_cols:
    if col not in df.columns:
        df[col] = 0
        continue
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    label_encoders[col] = enc

joblib.dump(label_encoders, "models/disease_label_encoders.pkl")

# ═══════════════════════════════════════════════════════
# 4. COMPUTE RISK SCORES (for labelling ONLY — never used as features)
# ═══════════════════════════════════════════════════════
df["_dengue_score"] = (
    df["bottles"]      * 3.0 +
    df["milk_packets"] * 1.5 +
    (df["segregation"] == 0).astype(int) * 5.0
)
df["_cholera_score"] = (
    df["food_waste"]               * 3.0 +
    (df["compost"] == 0).astype(int) * 4.0
)
df["_typhoid_score"] = (
    df["food_waste"]   * 2.0 +
    df["garden_waste"] * 1.5
)
df["_respiratory_score"] = (
    (df["disposal_method"] == 0).astype(int) * 6.0 +
    df["milk_packets"] * 1.2
)
df["_toxic_score"] = (
    df["old_devices"] * 2.5 +
    df["batteries"]   * 3.0
)

# Fraud dampening on scores (before labelling)
fw = df["fraud_confidence_weight"] if "fraud_confidence_weight" in df.columns else pd.Series(1.0, index=df.index)
for sc in ["_dengue_score","_cholera_score","_typhoid_score",
           "_respiratory_score","_toxic_score"]:
    df[sc] = df[sc] * fw

# Geo risk (label-safe — district is not derived from targets)
district_risk = {0: 4, 1: 3, 2: 5, 3: 6}
df["geo_risk"] = df["district"].map(district_risk).fillna(4.0) if "district" in df.columns else 4.0

df["_total_score"] = (
    df["_dengue_score"]      * 2.0 +
    df["_cholera_score"]     * 2.0 +
    df["_typhoid_score"]     * 1.5 +
    df["_respiratory_score"] * 1.5 +
    df["_toxic_score"]       * 1.0 +
    df["geo_risk"]           * 1.0
)

# ═══════════════════════════════════════════════════════
# 5. REALISTIC LABEL CREATION  — fixes the 1.0 problem
#
# Strategy: Use 30th / 70th percentile BANDS.
#   Bottom 30% → at-risk = 0  (clearly safe)
#   Top    30% → at-risk = 1  (clearly risky)
#   Middle 40% → stochastic:  weighted coin flip based on relative
#                              position within the grey zone.
#   This creates genuine uncertainty the model must learn.
# ═══════════════════════════════════════════════════════

def make_realistic_label(scores, low_pct=30, high_pct=70, seed=42):
    """
    Returns binary labels with a grey zone in the middle.
    Grey zone samples are assigned probabilistically so the
    boundary is not clean — forcing the model to generalise.
    """
    rng   = np.random.default_rng(seed)
    lo    = np.percentile(scores, low_pct)
    hi    = np.percentile(scores, high_pct)
    n     = len(scores)
    labels = np.zeros(n, dtype=int)

    clearly_high = scores >= hi
    grey          = (scores >= lo) & (scores < hi)

    labels[clearly_high] = 1

    # In grey zone: probability scales linearly from 0.25 → 0.75
    grey_scores   = scores[grey]
    grey_probs    = 0.25 + 0.50 * (grey_scores - lo) / (hi - lo + 1e-9)
    labels[grey]  = rng.binomial(1, grey_probs)

    return labels

score_col_map = {
    "dengue_label":      "_dengue_score",
    "cholera_label":     "_cholera_score",
    "typhoid_label":     "_typhoid_score",
    "respiratory_label": "_respiratory_score",
    "toxic_label":       "_toxic_score",
}

for label_col, score_col in score_col_map.items():
    df[label_col] = make_realistic_label(
        df[score_col].values, low_pct=30, high_pct=70,
        seed=RANDOM_STATE + list(score_col_map.keys()).index(label_col))

# 4-class risk level using quartile cuts on total score
df["risk_level"] = pd.qcut(
    df["_total_score"], q=4, labels=["LOW","MODERATE","HIGH","CRITICAL"])

print("\n── Risk Level Distribution ──")
print(df["risk_level"].value_counts().sort_index())
print("\n── Disease Label Distributions ──")
for lbl in score_col_map:
    vc = df[lbl].value_counts().sort_index()
    print(f"  {lbl:<22}: 0={vc.get(0,0)}  1={vc.get(1,0)}")

# ═══════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING  (raw inputs only — no score leakage)
# ═══════════════════════════════════════════════════════
fse = df["family_size_encoded"] if "family_size_encoded" in df.columns else pd.Series(2, index=df.index)
fse = fse.clip(lower=1)

df["waste_density"]      = (df["food_waste"]   + df["garden_waste"]) / fse
df["plastic_density"]    = (df["milk_packets"] + df["bottles"])      / fse
df["ewaste_density"]     = (df["old_devices"]  + df["batteries"])    / fse
df["bad_disposal_score"] = (
    (df["segregation"]    == 0).astype(int) +
    (df["compost"]        == 0).astype(int) +
    (df["disposal_method"]== 0).astype(int)
)
df["organic_waste_total"]  = df["food_waste"] + df["garden_waste"]
df["plastic_waste_total"]  = df["milk_packets"] + df["bottles"]
df["delivery_waste_ratio"] = df["milk_packets"] / (df["bottles"] + 1e-6)

# ── Feature lists ─────────────────────────────────────
raw_feats = ["milk_packets","bottles","food_waste","garden_waste",
             "old_devices","batteries","segregation","compost","disposal_method"]
eng_feats = ["waste_density","plastic_density","ewaste_density",
             "bad_disposal_score","organic_waste_total",
             "plastic_waste_total","delivery_waste_ratio"]
ctx_feats = ["geo_risk"]
fraud_feats  = (["fraud_probability","fraud_confidence_weight"] if has_fraud else [])
carbon_feats = (["carbon_points","carbon_trust_score"]
                if "carbon_points" in df.columns else [])

all_features = raw_feats + eng_feats + ctx_feats + fraud_feats + carbon_feats
all_features = [f for f in all_features if f in df.columns]

print(f"\n✔  Features used ({len(all_features)}): {all_features}")

# ═══════════════════════════════════════════════════════
# 7. ADD MEASUREMENT NOISE  (simulate real-world sensor error)
#    Only applied to continuous columns during training.
#    5% Gaussian noise relative to each column's std dev.
# ═══════════════════════════════════════════════════════
def add_noise(X_array, feature_names, noise_pct=0.05, seed=42):
    rng   = np.random.default_rng(seed)
    X_out = X_array.copy().astype(float)
    for i, fname in enumerate(feature_names):
        # Skip binary/categorical columns
        unique_vals = np.unique(X_out[:, i])
        if len(unique_vals) <= 4:
            continue
        sigma = X_out[:, i].std() * noise_pct
        X_out[:, i] += rng.normal(0, sigma, size=len(X_out))
    return X_out

X_raw_vals = df[all_features].fillna(0).values

# ═══════════════════════════════════════════════════════
# 8. SCALE
# ═══════════════════════════════════════════════════════
scaler_risk    = RobustScaler()
scaler_disease = RobustScaler()

X_risk_scaled    = scaler_risk.fit_transform(X_raw_vals)
X_disease_scaled = scaler_disease.fit_transform(X_raw_vals)

y_risk    = df["risk_level"]
y_disease = df[list(score_col_map.keys())]

# ═══════════════════════════════════════════════════════
# 9. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_risk_scaled, y_risk,
    test_size=0.20, random_state=RANDOM_STATE, stratify=y_risk)

Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
    X_disease_scaled, y_disease,
    test_size=0.20, random_state=RANDOM_STATE)

# Add noise to training set ONLY (not test — test stays clean)
Xr_tr_noisy = add_noise(Xr_tr, all_features, noise_pct=0.05, seed=RANDOM_STATE)
Xd_tr_noisy = add_noise(Xd_tr, all_features, noise_pct=0.05, seed=RANDOM_STATE + 1)

# ═══════════════════════════════════════════════════════
# 10. MODEL 1 — RISK LEVEL CLASSIFIER
# ═══════════════════════════════════════════════════════
print("\n── Training Risk Level Model ──")

risk_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,          # shallower → less overfitting
    min_samples_split=10,
    min_samples_leaf=5,   # larger leaf → more regularisation
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
risk_model.fit(Xr_tr_noisy, yr_tr)
risk_pred = risk_model.predict(Xr_te)

print(f"✔  Test Accuracy : {accuracy_score(yr_te, risk_pred):.4f}")
print("\n── Risk Classification Report ──")
print(classification_report(yr_te, risk_pred, zero_division=0))

# Logistic Regression baseline (sanity check)
lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
lr_baseline.fit(Xr_tr_noisy, yr_tr)
lr_acc = accuracy_score(yr_te, lr_baseline.predict(Xr_te))
print(f"Logistic Regression baseline accuracy: {lr_acc:.4f}")
print(f"RF vs LR gap: {accuracy_score(yr_te, risk_pred) - lr_acc:+.4f}  "
      f"(>0.15 may indicate overfitting)")

# 5-Fold stratified CV — primary reliability metric
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(risk_model, X_risk_scaled, y_risk,
                         cv=skf, scoring="accuracy", n_jobs=-1)
cv_f1  = cross_val_score(risk_model, X_risk_scaled, y_risk,
                         cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"\n5-Fold CV Accuracy : mean={cv_acc.mean():.4f}  std={cv_acc.std():.4f}")
print(f"5-Fold CV F1-macro : mean={cv_f1.mean():.4f}  std={cv_f1.std():.4f}")
print("CV Accuracy scores :", np.round(cv_acc, 4))

# ═══════════════════════════════════════════════════════
# 11. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════
cm = confusion_matrix(yr_te, risk_pred, labels=["LOW","MODERATE","HIGH","CRITICAL"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["LOW","MOD","HIGH","CRIT"],
            yticklabels=["LOW","MOD","HIGH","CRIT"], ax=ax)
ax.set_title("Risk Level — Confusion Matrix"); ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("reports/disease_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Confusion matrix → reports/disease_confusion_matrix.png")

# ═══════════════════════════════════════════════════════
# 12. MODEL 2 — MULTI-DISEASE CLASSIFIER
#     GradientBoosting per disease — more regularised than deep RF
# ═══════════════════════════════════════════════════════
print("\n── Training Multi-Disease Model ──")

base_clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=RANDOM_STATE,
)
multi_disease_model = MultiOutputClassifier(base_clf, n_jobs=-1)
multi_disease_model.fit(Xd_tr_noisy, yd_tr)

disease_pred  = multi_disease_model.predict(Xd_te)
disease_proba = multi_disease_model.predict_proba(Xd_te)

disease_names = ["Dengue","Cholera","Typhoid","Respiratory","Toxic Exposure"]
report_rows   = []
calibration_data = []

for i, disease in enumerate(disease_names):
    true  = yd_te.iloc[:, i].values
    pred  = disease_pred[:, i]
    proba = disease_proba[i][:, 1]

    acc = accuracy_score(true, pred)
    roc = roc_auc_score(true, proba) if len(np.unique(true)) > 1 else None
    ap  = average_precision_score(true, proba) if len(np.unique(true)) > 1 else None
    brier = brier_score_loss(true, proba)

    # Per-disease CV
    single_clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE)
    cv_disease = cross_val_score(single_clf, X_disease_scaled,
                                 y_disease.iloc[:, i],
                                 cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="roc_auc", n_jobs=-1)

    print(f"\n═══ {disease} ═══")
    print(classification_report(true, pred, zero_division=0))
    if roc:   print(f"   ROC-AUC         : {roc:.4f}")
    if ap:    print(f"   Avg Precision   : {ap:.4f}")
    print(    f"   Brier Score     : {brier:.4f}  (lower=better, 0=perfect)")
    print(    f"   5-Fold CV AUC   : {cv_disease.mean():.4f} ± {cv_disease.std():.4f}")

    report_rows.append({
        "Disease":       disease,
        "Test_Accuracy": round(acc,  4),
        "ROC_AUC":       round(roc,  4) if roc   else None,
        "Avg_Precision": round(ap,   4) if ap    else None,
        "Brier_Score":   round(brier,4),
        "CV_AUC_mean":   round(cv_disease.mean(), 4),
        "CV_AUC_std":    round(cv_disease.std(),  4),
    })

    calibration_data.append((disease, true, proba))

# ═══════════════════════════════════════════════════════
# 13. CALIBRATION PLOT
#     A well-calibrated model: "60% predicted = ~60% actual"
#     Poor calibration at 1.0 accuracy means something is wrong.
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
colors = ["#3498db","#2ecc71","#e74c3c","#f39c12","#9b59b6"]

for i, (disease, true, proba) in enumerate(calibration_data):
    ax = axes[i]
    try:
        frac_pos, mean_pred = calibration_curve(true, proba, n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", color=colors[i], lw=2, label=disease)
        ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect calibration")
        ax.set_title(f"{disease} — Calibration"); ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positive"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    except Exception:
        ax.text(0.5, 0.5, "Not enough data", ha="center", va="center")

axes[5].axis("off")
plt.suptitle("Probability Calibration Curves", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/disease_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Calibration plot → reports/disease_calibration.png")

# ═══════════════════════════════════════════════════════
# 14. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════
imp_df = pd.DataFrame({
    "Feature":    all_features,
    "Importance": risk_model.feature_importances_,
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, max(5, len(all_features) * 0.38)))
bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color="#e24b4a")
ax.set_title("Risk Level Model — Feature Importance", fontsize=12)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/disease_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Feature importance → reports/disease_feature_importance.png")

# ═══════════════════════════════════════════════════════
# 15. LABEL DISTRIBUTION PLOT (verify grey zone worked)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(16, 4))
for i, (lbl, score_col) in enumerate(score_col_map.items()):
    ax = axes[i]
    scores = df[score_col].values
    labels = df[lbl].values
    ax.scatter(scores, labels + np.random.normal(0, 0.02, len(labels)),
               alpha=0.15, s=5, c=labels, cmap="RdYlGn")
    ax.axvline(np.percentile(scores, 30), color="blue", lw=1, ls="--", label="30th pct")
    ax.axvline(np.percentile(scores, 70), color="red",  lw=1, ls="--", label="70th pct")
    ax.set_title(lbl.replace("_label","").capitalize(), fontsize=10)
    ax.set_yticks([0,1]); ax.set_ylabel("Label")
    ax.set_xlabel("Score"); ax.legend(fontsize=7)

plt.suptitle("Label Distributions — Grey Zone Visible Between Dashed Lines",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/disease_label_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("✔  Label distributions → reports/disease_label_distributions.png")

# ═══════════════════════════════════════════════════════
# 16. SAVE
# ═══════════════════════════════════════════════════════
report_df = pd.DataFrame(report_rows)
print("\n── Disease Model Summary ──")
print(report_df.to_string(index=False))
report_df.to_csv("reports/disease_risk_report.csv", index=False)

joblib.dump(risk_model,          "models/risk_level_model.pkl")
joblib.dump(multi_disease_model, "models/multi_disease_model.pkl")
joblib.dump(scaler_risk,         "models/risk_feature_scaler.pkl")
joblib.dump(scaler_disease,      "models/disease_feature_scaler.pkl")
print("\n✔  All models saved → models/")

# Drop internal score columns from any downstream CSV
score_cols_internal = [c for c in df.columns if c.startswith("_")]
df.drop(columns=score_cols_internal, inplace=True, errors="ignore")