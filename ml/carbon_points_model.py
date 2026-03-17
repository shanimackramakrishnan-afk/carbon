"""
carbon_points_model.py
======================
Tiered carbon scoring with fraud-aware trust score, eco-star bonus,
and partial credit for fraud households with genuine green behaviour.

Reads  : datasets/fraud_scored_full.csv  (output of ai_fraud_detection.py)
Outputs: output/final_household_carbon_scores.csv
         output/clean_household_carbon_scores.csv
         reports/carbon_level_summary.csv
"""

import os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.makedirs("output",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD (uses fraud-scored dataset)
# ─────────────────────────────────────────────
df = pd.read_csv("datasets/fraud_scored_full.csv")
print(f"✔  Loaded {len(df):,} rows")

# ─────────────────────────────────────────────
# 2. CARBON POINTS ENGINE
# ─────────────────────────────────────────────
df["carbon_points"] = 50.0

# ── Green Rewards ─────────────────────────────
df["carbon_points"] += pd.to_numeric(df["bottles"], errors="coerce").fillna(0) * 3
df.loc[df["compost"]        == "yes",        "carbon_points"] += 15
df.loc[df["segregation"]    == "yes",        "carbon_points"] += 20
df.loc[df["disposal_method"]== "recycling",  "carbon_points"] += 15
df.loc[df["batteries"]      == "recycle",    "carbon_points"] += 10
df.loc[df["old_devices"]    == "recycle",    "carbon_points"] += 10

# ── Eco-Star Bonus (+20%) for clean triple-green households ───
eco_star = (
    (df["segregation"]    == "yes") &
    (df["compost"]        == "yes") &
    (df["disposal_method"]== "recycling") &
    (df["fraud_flag"]     == 0)
)
df.loc[eco_star, "carbon_points"] *= 1.20
df["eco_star"] = eco_star.astype(int)

# ── Consumption Penalties ─────────────────────
df["carbon_points"] -= pd.to_numeric(df["milk_packets"],  errors="coerce").fillna(0) * 1.5
df["carbon_points"] -= pd.to_numeric(df["deliveries"],    errors="coerce").fillna(0) * 1.0
df["carbon_points"] -= pd.to_numeric(df["food_waste"],    errors="coerce").fillna(0) * 2.0
df["carbon_points"] -= pd.to_numeric(df["garden_waste"],  errors="coerce").fillna(0) * 1.0
df.loc[df["segregation"] == "no", "carbon_points"] -= 15
df.loc[df["compost"]     == "no", "carbon_points"] -= 5

# ── Fraud Tiered Penalties ────────────────────
# Fraud households still earn partial credit for genuine actions.
# Penalty applied AFTER rewards so green behaviour is not erased.
penalty_map = {"SEVERE": 50, "MODERATE": 35, "MINOR": 20, "NONE": 0}
df["fraud_penalty"] = df["fraud_severity"].map(penalty_map).fillna(0)
df["carbon_points"] -= df["fraud_penalty"]

# ── Floor & Normalise to 0-100 ────────────────
df["carbon_points"] = df["carbon_points"].clip(lower=0)
cp_min, cp_max = df["carbon_points"].min(), df["carbon_points"].max()
df["carbon_points"] = ((df["carbon_points"] - cp_min) / (cp_max - cp_min) * 100).round(2)

# ── Carbon Trust Score (fraud households capped at 40) ────────
df["carbon_trust_score"] = df["carbon_points"].copy()
df.loc[df["fraud_flag"] == 1, "carbon_trust_score"] = (
    df.loc[df["fraud_flag"] == 1, "carbon_trust_score"].clip(upper=40)
)
df["carbon_trust_score"] = df["carbon_trust_score"].round(2)

# ─────────────────────────────────────────────
# 3. CARBON LEVEL LABELS
# ─────────────────────────────────────────────
def carbon_level(score, sev):
    if sev == "SEVERE":   return "FRAUD-SEVERE"
    if sev == "MODERATE": return "FRAUD-MODERATE"
    if sev == "MINOR":    return "FRAUD-MINOR"
    if score >= 80: return "EXCELLENT"
    if score >= 60: return "GOOD"
    if score >= 40: return "AVERAGE"
    if score >= 20: return "POOR"
    return "CRITICAL"

df["carbon_level"] = df.apply(
    lambda r: carbon_level(r["carbon_trust_score"], r["fraud_severity"]), axis=1)

# ─────────────────────────────────────────────
# 4. SUMMARY STATS
# ─────────────────────────────────────────────
print("\n── Carbon Level Distribution ──")
print(df["carbon_level"].value_counts().to_string())

fraud_df  = df[df["fraud_flag"] == 1]
clean_df  = df[df["fraud_flag"] == 0]
eco_df    = df[df["eco_star"]   == 1]

print(f"\nAvg trust score — all    : {df['carbon_trust_score'].mean():.2f}")
print(f"Avg trust score — clean  : {clean_df['carbon_trust_score'].mean():.2f}")
print(f"Avg trust score — fraud  : {fraud_df['carbon_trust_score'].mean():.2f}")
print(f"Eco-star households      : {df['eco_star'].sum()}")

# ─────────────────────────────────────────────
# 5. PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Carbon Trust Score Overview", fontsize=14, fontweight="bold")

ax = axes[0]
ax.hist(clean_df["carbon_trust_score"], bins=30, alpha=0.6, color="#378add", label="Clean")
ax.hist(fraud_df["carbon_trust_score"], bins=30, alpha=0.6, color="#e24b4a", label="Fraud")
ax.set_title("Carbon Trust Score Distribution")
ax.set_xlabel("Score (0-100)"); ax.legend()

ax = axes[1]
order = ["EXCELLENT","GOOD","AVERAGE","POOR","CRITICAL",
         "FRAUD-MINOR","FRAUD-MODERATE","FRAUD-SEVERE"]
colors = ["#27ae60","#2ecc71","#f1c40f","#e67e22","#e74c3c",
          "#85b7eb","#e24b4a","#a32d2d"]
counts = df["carbon_level"].value_counts().reindex(order, fill_value=0)
bars = ax.barh(counts.index, counts.values, color=colors)
ax.set_title("Carbon Level Breakdown")
ax.set_xlabel("Households")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("reports/carbon_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✔  Plot saved → reports/carbon_overview.png")

# ─────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────
df.to_csv("output/final_household_carbon_scores.csv", index=False)
clean_df.to_csv("output/clean_household_carbon_scores.csv", index=False)

summary = df.groupby("carbon_level").agg(
    count=("carbon_trust_score","count"),
    avg_trust=("carbon_trust_score","mean"),
    avg_fraud_prob=("fraud_probability","mean"),
).round(2)
summary.to_csv("reports/carbon_level_summary.csv")

print("✔  Outputs saved → output/ and reports/")