"""
app.py  —  Waste Management + ML Intelligence Platform
=======================================================
Integrates all ML models into the Flask web app:
  • Fraud detection    (IsolationForest ensemble)
  • Carbon scoring     (tiered reward/penalty engine)
  • Waste estimation   (bio / plastic / e-waste)
  • Disease risk       (risk level + 5-disease predictor)

Flow:
  register → login → plastic form → biodegradable form
  → ewaste form → [ML pipeline runs] → results dashboard

New routes vs original:
  /results   — full ML results dashboard
  /history   — user's past 10 submissions
  /api/results — JSON endpoint for results
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
import numpy as np
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "waste_ml_secret_2024"

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
# Auto-detect models folder — checks multiple possible locations
def _find_model_dir():
    candidates = [
        "models",
        "ml/models",
        "carbon/models",
        "carbon/ml/models",
        os.path.join(os.path.dirname(__file__), "models"),
        os.path.join(os.path.dirname(__file__), "ml", "models"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            # Verify it actually contains pkl files
            pkls = [f for f in os.listdir(path) if f.endswith(".pkl")]
            if pkls:
                print(f"  Found models folder: {path}  ({len(pkls)} .pkl files)")
                return path
    print("  WARNING: No models folder found. Checked:", candidates)
    return "models"  # fallback default

MODEL_DIR = _find_model_dir()

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    print(f"  WARNING: model not found: {path}")
    return None

fraud_model     = load_model("fraud_detection_model.pkl")
robust_scaler   = load_model("robust_scaler.pkl")
family_encoder  = load_model("family_encoder.pkl")
battery_encoder = load_model("battery_encoder.pkl")
risk_model      = load_model("risk_level_model.pkl")
disease_model   = load_model("multi_disease_model.pkl")
disease_scaler  = load_model("disease_feature_scaler.pkl")
label_encoders  = load_model("disease_label_encoders.pkl")

models_ready = all([fraud_model, robust_scaler, risk_model, disease_model])
print("=" * 55)
print("MODEL LOAD STATUS")
print("=" * 55)
print(f"  fraud_model     : {'OK' if fraud_model    else 'MISSING'}")
print(f"  robust_scaler   : {'OK' if robust_scaler  else 'MISSING'}")
print(f"  family_encoder  : {'OK' if family_encoder else 'MISSING'}")
print(f"  battery_encoder : {'OK' if battery_encoder else 'MISSING'}")
print(f"  risk_model      : {'OK' if risk_model     else 'MISSING'}")
print(f"  disease_model   : {'OK' if disease_model  else 'MISSING'}")
print(f"  disease_scaler  : {'OK' if disease_scaler else 'MISSING'}")
if disease_scaler:
    print(f"  disease_scaler expects {disease_scaler.n_features_in_} features")
if robust_scaler:
    print(f"  fraud_scaler   expects {robust_scaler.n_features_in_} features")
print("=" * 55)

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
def get_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/user.db")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, email TEXT UNIQUE, password TEXT,
        aadhaar TEXT UNIQUE, created_at TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS plastic_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT, family_size TEXT, house_type TEXT,
        resident_name TEXT, house_no TEXT, apartment_name TEXT,
        flat_no TEXT, street TEXT, landmark TEXT,
        ward_number TEXT, local_body TEXT, district TEXT, pincode TEXT,
        milk_packets INTEGER, deliveries INTEGER, oil_type TEXT,
        bottles TEXT, segregation TEXT, hks_frequency TEXT,
        receipt TEXT, date_time TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS biodegradable_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT, food_waste TEXT, compost TEXT,
        garden_waste TEXT, date_time TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS ewaste_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT, old_devices TEXT, batteries TEXT,
        disposal_method TEXT, date_time TEXT)""")

    conn.execute("""CREATE TABLE IF NOT EXISTS ml_results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        fraud_probability REAL, fraud_flag INTEGER, fraud_severity TEXT,
        carbon_points REAL, carbon_trust_score REAL, carbon_level TEXT,
        bio_waste REAL, plastic_waste REAL, e_waste REAL,
        bio_1m REAL, bio_3m REAL, bio_12m REAL,
        plastic_1m REAL, plastic_3m REAL, plastic_12m REAL,
        ewaste_1m REAL, ewaste_3m REAL, ewaste_12m REAL,
        risk_level TEXT,
        dengue_risk INTEGER, dengue_prob REAL,
        cholera_risk INTEGER, cholera_prob REAL,
        typhoid_risk INTEGER, typhoid_prob REAL,
        respiratory_risk INTEGER, respiratory_prob REAL,
        toxic_risk INTEGER, toxic_prob REAL,
        date_time TEXT)""")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────
FRAUD_THRESH = 70.0

def run_ml_pipeline(user_email, conn):
    """
    Pull latest form data for a user, run all ML models,
    persist results to ml_results table, return results dict.
    """
    plastic = conn.execute(
        "SELECT * FROM plastic_data WHERE user_email=? ORDER BY id DESC LIMIT 1",
        (user_email,)).fetchone()
    bio = conn.execute(
        "SELECT * FROM biodegradable_data WHERE user_email=? ORDER BY id DESC LIMIT 1",
        (user_email,)).fetchone()
    ew = conn.execute(
        "SELECT * FROM ewaste_data WHERE user_email=? ORDER BY id DESC LIMIT 1",
        (user_email,)).fetchone()

    if not (plastic and bio and ew):
        return None

    # ── Parse ────────────────────────────────
    def si(v, d=0):
        try: return int(v)
        except: return d

    def sf(v, d=0.0):
        try: return float(v)
        except: return d

    milk_packets  = si(plastic["milk_packets"])
    deliveries    = si(plastic["deliveries"])
    bottles       = si(plastic["bottles"])
    family_size   = str(plastic["family_size"])
    district      = str(plastic["district"])
    segregation   = str(plastic["segregation"])

    food_waste    = sf(bio["food_waste"])
    compost       = str(bio["compost"])
    garden_waste  = sf(bio["garden_waste"])

    old_devices   = si(ew["old_devices"])
    batteries_str = str(ew["batteries"])
    batteries_num = sf(ew["batteries"])
    disposal      = str(ew["disposal_method"])

    # ── Encode ────────────────────────────────
    fam_enc = 0
    bat_enc = 0
    try:
        fam_enc = int(family_encoder.transform([family_size])[0]) if family_encoder else 0
    except: pass
    try:
        bat_enc = int(battery_encoder.transform([batteries_str])[0]) if battery_encoder else 0
    except: pass

    fam_div = max(fam_enc, 1)
    eps     = 1e-6

    milk_pp    = milk_packets  / fam_div
    del_pp     = deliveries    / fam_div
    dev_pp     = old_devices   / fam_div
    milk_x_del = milk_packets  * deliveries
    waste_pp   = (food_waste + garden_waste) / fam_div

    # Approximate z-scores using population norms
    milk_z = abs((milk_packets - 8)  / 4)
    del_z  = abs((deliveries   - 5)  / 3)
    dev_z  = abs((old_devices  - 2)  / 2)
    fw_z   = abs((food_waste   - 3)  / 2)

    # ─────────────────────────────────────────
    # 1. FRAUD DETECTION
    # ─────────────────────────────────────────
    fraud_prob = 0.0
    fraud_flag = 0
    fraud_sev  = "NONE"

    if fraud_model and robust_scaler:
        try:
            X_f = robust_scaler.transform([[
                fam_enc, milk_packets, deliveries, old_devices, bat_enc,
                milk_pp, del_pp, dev_pp, milk_x_del, waste_pp,
                milk_z, del_z, dev_z, fw_z,
            ]])
            score      = fraud_model.decision_function(X_f)[0]
            fraud_prob = round(float(np.clip((0.5 - score) / 1.0, 0, 1)) * 100, 2)
            fraud_flag = 1 if fraud_prob > FRAUD_THRESH else 0
            if fraud_flag:
                fraud_sev = ("SEVERE"   if fraud_prob > 90 else
                             "MODERATE" if fraud_prob > 80 else "MINOR")
        except Exception as e:
            print(f"Fraud model error: {e}")

    # ─────────────────────────────────────────
    # 2. CARBON SCORING
    # ─────────────────────────────────────────
    pts = 50.0
    pts += bottles * 3
    if compost     == "yes":        pts += 15
    if segregation == "yes":        pts += 20
    if disposal    == "recycling":  pts += 15
    if batteries_str == "recycle":  pts += 10
    if str(ew["old_devices"]) == "recycle": pts += 10

    eco_star = (compost=="yes" and segregation=="yes"
                and disposal=="recycling" and fraud_flag==0)
    if eco_star: pts *= 1.20

    pts -= milk_packets * 1.5
    pts -= deliveries   * 1.0
    pts -= food_waste   * 2.0
    pts -= garden_waste * 1.0
    if segregation == "no": pts -= 15
    if compost     == "no": pts -= 5

    pen = {"SEVERE":50,"MODERATE":35,"MINOR":20,"NONE":0}.get(fraud_sev,0)
    pts  = max(0, pts - pen)

    c_norm  = round(min(100, (pts / 180) * 100), 2)
    c_trust = round(c_norm if fraud_flag == 0 else min(40, c_norm), 2)

    def c_level(s, sv):
        if sv=="SEVERE":   return "FRAUD-SEVERE"
        if sv=="MODERATE": return "FRAUD-MODERATE"
        if sv=="MINOR":    return "FRAUD-MINOR"
        if s>=80: return "EXCELLENT"
        if s>=60: return "GOOD"
        if s>=40: return "AVERAGE"
        if s>=20: return "POOR"
        return "CRITICAL"

    carbon_level = c_level(c_trust, fraud_sev)

    # ─────────────────────────────────────────
    # 3. WASTE ESTIMATION + PROJECTIONS
    # ─────────────────────────────────────────
    bio_raw     = food_waste * 0.65 + garden_waste * 0.42 + fam_div * 0.18
    plastic_raw = milk_packets * 0.38 + bottles * 0.44 + deliveries * 0.22
    ew_raw      = old_devices * 0.72 + batteries_num * 0.28

    fw_map = {"SEVERE":0.10,"MODERATE":0.40,"MINOR":0.70,"NONE":1.0}
    fw     = max(fw_map.get(fraud_sev,1.0) * (1 - fraud_prob/100), 0.05)

    bio_adj  = round(bio_raw     * fw, 3)
    pla_adj  = round(plastic_raw * fw, 3)
    ew_adj   = round(ew_raw      * fw, 3)

    # Seasonal 12-month projections
    bio_seasonal = [1.0,1.1,1.2,1.05,0.95,0.90,1.0,1.0,0.95,1.0,1.1,1.15]
    pla_seasonal = [1.0,1.0,1.0,1.0,1.05,1.05,1.1,1.05,1.0,1.0,1.1,1.2]
    ew_seasonal  = [0.9,0.9,1.0,1.0,1.0,1.0,0.9,0.9,1.0,1.1,1.2,1.4]

    bio_12  = [round(bio_adj  * s, 4) for s in bio_seasonal]
    pla_12  = [round(pla_adj  * s, 4) for s in pla_seasonal]
    ew_12   = [round(ew_adj   * s, 4) for s in ew_seasonal]

    # ─────────────────────────────────────────
    # 4. DISEASE RISK
    # ─────────────────────────────────────────
    risk_lvl  = "UNKNOWN"
    dis_preds  = {k: 0   for k in ["dengue","cholera","typhoid","respiratory","toxic"]}
    dis_probas = {k: 0.0 for k in ["dengue","cholera","typhoid","respiratory","toxic"]}

    seg_enc  = 1 if segregation == "yes"       else 0
    comp_enc = 1 if compost     == "yes"       else 0
    disp_enc = 1 if disposal    == "recycling" else 0

    geo = {"Thiruvananthapuram":6,"Ernakulam":5,"Kozhikode":4,"Thrissur":3}.get(district, 4)

    waste_d = (food_waste   + garden_waste)  / fam_div
    plas_d  = (milk_packets + bottles)       / fam_div
    ew_d    = (old_devices  + batteries_num) / fam_div
    bad_d   = (1 - seg_enc) + (1 - comp_enc) + (1 - disp_enc)
    org_tot = food_waste   + garden_waste
    pla_tot = milk_packets + bottles
    del_rat = milk_packets / (bottles + eps)

    # Feature order matches disease_prediction.py exactly:
    # raw(9) + engineered(7) + geo(1) + fraud(2) + carbon(2) = 21
    base_inp = [
        milk_packets, bottles, food_waste, garden_waste,
        old_devices, batteries_num,
        seg_enc, comp_enc, disp_enc,
        waste_d, plas_d, ew_d, bad_d,
        org_tot, pla_tot, del_rat,
        geo,
        fraud_prob, fw,
        c_norm, c_trust,
    ]

    if disease_scaler:
        try:
            n   = disease_scaler.n_features_in_
            inp = base_inp[:n] if len(base_inp) >= n else base_inp + [0] * (n - len(base_inp))
            X_d = disease_scaler.transform([inp])

            if risk_model:
                risk_lvl = str(risk_model.predict(X_d)[0])

            if disease_model:
                preds  = disease_model.predict(X_d)[0]
                probas = disease_model.predict_proba(X_d)
                keys   = ["dengue","cholera","typhoid","respiratory","toxic"]
                for i, k in enumerate(keys):
                    dis_preds[k] = int(preds[i])
                    try:
                        # probas[i] shape = (1, 2) for each output
                        dis_probas[k] = round(float(probas[i][0][1]), 3)
                    except (IndexError, TypeError):
                        try:
                            dis_probas[k] = round(float(probas[i][1]), 3)
                        except Exception:
                            dis_probas[k] = float(dis_preds[k])

            print(f"Disease OK  risk={risk_lvl}  preds={dis_preds}  probas={dis_probas}")

        except Exception as e:
            import traceback
            print(f"Disease model error: {e}")
            traceback.print_exc()

    # ─────────────────────────────────────────
    # 5. PERSIST TO DB
    # ─────────────────────────────────────────
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
    INSERT INTO ml_results(
        user_email,
        fraud_probability, fraud_flag, fraud_severity,
        carbon_points, carbon_trust_score, carbon_level,
        bio_waste, plastic_waste, e_waste,
        bio_1m, bio_3m, bio_12m,
        plastic_1m, plastic_3m, plastic_12m,
        ewaste_1m, ewaste_3m, ewaste_12m,
        risk_level,
        dengue_risk, dengue_prob,
        cholera_risk, cholera_prob,
        typhoid_risk, typhoid_prob,
        respiratory_risk, respiratory_prob,
        toxic_risk, toxic_prob,
        date_time
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_email,
        fraud_prob, fraud_flag, fraud_sev,
        c_norm, c_trust, carbon_level,
        bio_adj, pla_adj, ew_adj,
        bio_adj, round(sum(bio_12[:3]),3),  round(sum(bio_12),3),
        pla_adj, round(sum(pla_12[:3]),3),  round(sum(pla_12),3),
        ew_adj,  round(sum(ew_12[:3]),3),   round(sum(ew_12),3),
        risk_lvl,
        dis_preds["dengue"],       dis_probas["dengue"],
        dis_preds["cholera"],      dis_probas["cholera"],
        dis_preds["typhoid"],      dis_probas["typhoid"],
        dis_preds["respiratory"],  dis_probas["respiratory"],
        dis_preds["toxic"],        dis_probas["toxic"],
        dt,
    ))
    conn.commit()

    return {
        "fraud":   {"probability": fraud_prob, "flag": fraud_flag, "severity": fraud_sev},
        "carbon":  {"points": c_norm, "trust_score": c_trust,
                    "level": carbon_level, "eco_star": eco_star},
        "waste":   {
            "bio": bio_adj, "plastic": pla_adj, "ewaste": ew_adj,
            "bio_3m":  round(sum(bio_12[:3]),3),
            "plastic_3m": round(sum(pla_12[:3]),3),
            "ewaste_3m":  round(sum(ew_12[:3]),3),
            "bio_12m":    round(sum(bio_12),3),
            "plastic_12m":round(sum(pla_12),3),
            "ewaste_12m": round(sum(ew_12),3),
            "monthly_bio":     bio_12,
            "monthly_plastic": pla_12,
            "monthly_ewaste":  ew_12,
        },
        "disease": {
            "risk_level":    risk_lvl,
            "predictions":   dis_preds,
            "probabilities": dis_probas,
        },
        "submitted_at": dt,
    }


# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/register_submit", methods=["POST"])
def register_submit():
    name     = request.form["name"]
    email    = request.form["email"]
    password = request.form["password"]
    aadhaar  = request.form["aadhaar"]
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (name,email,password,aadhaar,created_at) VALUES (?,?,?,?,?)",
            (name, email, password, aadhaar, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    except:
        conn.close()
        return render_template("error.html", message="Email or Aadhaar already registered.")
    conn.close()
    return redirect(url_for("login"))


@app.route("/login_submit", methods=["POST"])
def login_submit():
    email    = request.form["email"]
    password = request.form["password"]
    aadhaar  = request.form["aadhaar"]
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=? AND aadhaar=?",
        (email, password, aadhaar)).fetchone()
    conn.close()
    if user:
        session["user"] = email
        session["name"] = user["name"]
        return redirect(url_for("plastic"))
    return render_template("error.html", message="Invalid credentials. Please try again.")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/plastic")
def plastic():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("plastic.html")


@app.route("/plastic_submit", methods=["POST"])
def plastic_submit():
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    conn.execute("""
    INSERT INTO plastic_data(
        user_email,family_size,house_type,resident_name,
        house_no,apartment_name,flat_no,street,landmark,
        ward_number,local_body,district,pincode,
        milk_packets,deliveries,oil_type,bottles,
        segregation,hks_frequency,receipt,date_time
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        session["user"],
        request.form["family_size"], request.form["house_type"],
        request.form["resident_name"], request.form["house_no"],
        request.form.get("apartment_name",""), request.form.get("flat_no",""),
        request.form["street"], request.form.get("landmark",""),
        request.form["ward_number"], request.form["local_body"],
        request.form["district"], request.form["pincode"],
        request.form["milk_packets"], request.form["deliveries"],
        request.form["oil_type"], request.form["bottles"],
        request.form["segregation"], request.form["hks_frequency"],
        request.form["receipt"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()
    return redirect(url_for("biodegradable"))


@app.route("/biodegradable")
def biodegradable():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("biodegradable.html")


@app.route("/biodegradable_submit", methods=["POST"])
def biodegradable_submit():
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    conn.execute("""
    INSERT INTO biodegradable_data(user_email,food_waste,compost,garden_waste,date_time)
    VALUES (?,?,?,?,?)
    """, (
        session["user"],
        request.form["food_waste"], request.form["compost"],
        request.form["garden_waste"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()
    return redirect(url_for("ewaste"))


@app.route("/ewaste")
def ewaste():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("ewaste.html")


@app.route("/ewaste_submit", methods=["POST"])
def ewaste_submit():
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    conn.execute("""
    INSERT INTO ewaste_data(user_email,old_devices,batteries,disposal_method,date_time)
    VALUES (?,?,?,?,?)
    """, (
        session["user"],
        request.form["old_devices"], request.form["batteries"],
        request.form["disposal_method"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()

    # ── All 3 forms done — run ML now ─────────
    results = run_ml_pipeline(session["user"], conn)
    conn.close()

    if results:
        session["ml_results"] = results
        return redirect(url_for("results"))
    return redirect(url_for("success"))


# ─────────────────────────────────────────────
# RESULTS DASHBOARD
# ─────────────────────────────────────────────
@app.route("/results")
def results():
    if "user" not in session:
        return redirect(url_for("login"))
    ml = session.get("ml_results")
    if not ml:
        conn = get_db()
        row  = conn.execute(
            "SELECT * FROM ml_results WHERE user_email=? ORDER BY id DESC LIMIT 1",
            (session["user"],)).fetchone()
        conn.close()
        ml = dict(row) if row else None
    if not ml:
        return redirect(url_for("plastic"))
    return render_template("results.html",
                           ml=ml,
                           user_name=session.get("name",""),
                           user_email=session.get("user",""))


# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM ml_results WHERE user_email=? ORDER BY id DESC LIMIT 10",
        (session["user"],)).fetchall()
    conn.close()
    return render_template("history.html",
                           records=[dict(r) for r in rows],
                           user_name=session.get("name",""))


# ─────────────────────────────────────────────
# SUCCESS (fallback if ML not available)
# ─────────────────────────────────────────────
@app.route("/success")
def success():
    return render_template("success.html",
                           time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ─────────────────────────────────────────────
# JSON API
# ─────────────────────────────────────────────
@app.route("/api/results")
def api_results():
    if "user" not in session:
        return jsonify({"error": "not logged in"}), 401
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM ml_results WHERE user_email=? ORDER BY id DESC LIMIT 1",
        (session["user"],)).fetchone()
    conn.close()
    return jsonify(dict(row)) if row else jsonify({"error": "no results"}), 404


@app.route("/api/history")
def api_history():
    if "user" not in session:
        return jsonify({"error": "not logged in"}), 401
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM ml_results WHERE user_email=? ORDER BY id DESC LIMIT 20",
        (session["user"],)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])




# ─────────────────────────────────────────────
# DEBUG ROUTE — visit /debug after logging in
# Shows exactly what models loaded and feature counts
# REMOVE THIS before going to production
# ─────────────────────────────────────────────
@app.route("/debug")
def debug():
    import json

    info = {}

    # Model load status
    info["models"] = {
        "fraud_model":     str(type(fraud_model)),
        "robust_scaler":   str(type(robust_scaler)),
        "risk_model":      str(type(risk_model)),
        "disease_model":   str(type(disease_model)),
        "disease_scaler":  str(type(disease_scaler)),
        "family_encoder":  str(type(family_encoder)),
        "battery_encoder": str(type(battery_encoder)),
    }

    # Feature counts
    info["feature_counts"] = {}
    if robust_scaler:
        info["feature_counts"]["fraud_scaler_expects"] = robust_scaler.n_features_in_
    if disease_scaler:
        info["feature_counts"]["disease_scaler_expects"] = disease_scaler.n_features_in_
    if risk_model:
        try:
            info["feature_counts"]["risk_model_expects"] = risk_model.n_features_in_
        except Exception:
            info["feature_counts"]["risk_model_expects"] = "unknown"
    if disease_model:
        try:
            info["feature_counts"]["disease_model_outputs"] = len(disease_model.estimators_)
        except Exception:
            info["feature_counts"]["disease_model_outputs"] = "unknown"

    # App sends 21 features to disease model — show them labelled
    info["app_sends_21_features"] = [
        "milk_packets", "bottles", "food_waste", "garden_waste",
        "old_devices", "batteries",
        "segregation_enc", "compost_enc", "disposal_enc",
        "waste_density", "plastic_density", "ewaste_density", "bad_disposal_score",
        "organic_waste_total", "plastic_waste_total", "delivery_waste_ratio",
        "geo_risk",
        "fraud_probability", "fraud_confidence_weight",
        "carbon_points", "carbon_trust_score",
    ]

    # Test with dummy data if user is logged in
    if "user" in session and disease_scaler:
        try:
            dummy = [5, 3, 2.0, 1.0, 1, 2, 1, 1, 1,
                     1.5, 2.0, 0.5, 0, 3.0, 8.0, 1.67, 4,
                     10.0, 0.9, 65.0, 65.0]
            n   = disease_scaler.n_features_in_
            inp = dummy[:n] if len(dummy) >= n else dummy + [0]*(n-len(dummy))
            X_d = disease_scaler.transform([inp])
            if risk_model:
                info["test_risk_prediction"] = str(risk_model.predict(X_d)[0])
            if disease_model:
                preds  = disease_model.predict(X_d)[0]
                probas = disease_model.predict_proba(X_d)
                keys   = ["dengue","cholera","typhoid","respiratory","toxic"]
                test_result = {}
                for i, k in enumerate(keys):
                    try:
                        p = round(float(probas[i][0][1]), 3)
                    except Exception:
                        p = 0.0
                    test_result[k] = {"pred": int(preds[i]), "prob": p}
                info["test_disease_predictions"] = test_result
        except Exception as e:
            import traceback
            info["test_error"] = str(e)
            info["test_traceback"] = traceback.format_exc()

    html = "<pre style=\'background:#111;color:#3ddc84;padding:24px;font-size:13px;\'>"
    html += json.dumps(info, indent=2)
    html += "</pre>"
    return html

if __name__ == "__main__":
    init_db()
    app.run(debug=True)