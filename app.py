from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "waste_secret_key"


# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    if not os.path.exists("database"):
        os.makedirs("database")

    conn = sqlite3.connect("database/user.db")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- CREATE TABLES ----------------
def init_db():
    conn = get_db_connection()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        aadhaar TEXT UNIQUE,
        created_at TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS plastic_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        family_size TEXT,
        house_type TEXT,
        resident_name TEXT,
        house_no TEXT,
        apartment_name TEXT,
        flat_no TEXT,
        street TEXT,
        landmark TEXT,
        ward_number TEXT,
        local_body TEXT,
        district TEXT,
        pincode TEXT,
        milk_packets INTEGER,
        deliveries INTEGER,
        oil_type TEXT,
        bottles TEXT,
        segregation TEXT,
        hks_frequency TEXT,
        receipt TEXT,
        date_time TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS biodegradable_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        food_waste TEXT,
        compost TEXT,
        garden_waste TEXT,
        date_time TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS ewaste_data(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        old_devices TEXT,
        batteries TEXT,
        disposal_method TEXT,
        date_time TEXT
    )
    """)

    conn.commit()
    conn.close()


# ---------------- HOME / LOGIN ----------------
@app.route("/")
def login():
    return render_template("login.html")


# ---------------- REGISTER PAGE ----------------
@app.route("/register")
def register():
    return render_template("register.html")


# ---------------- REGISTER SUBMIT ----------------
@app.route("/register_submit", methods=["POST"])
def register_submit():

    name = request.form["name"]
    email = request.form["email"]
    password = request.form["password"]
    aadhaar = request.form["aadhaar"]

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()

    try:
        conn.execute(
            "INSERT INTO users (name,email,password,aadhaar,created_at) VALUES (?,?,?,?,?)",
            (name,email,password,aadhaar,created_at)
        )
        conn.commit()
    except:
        conn.close()
        return "User already exists"

    conn.close()
    return redirect(url_for("login"))


# ---------------- LOGIN ----------------
@app.route("/login_submit", methods=["POST"])
def login_submit():

    email = request.form["email"]
    password = request.form["password"]
    aadhaar = request.form["aadhaar"]

    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=? AND aadhaar=?",
        (email,password,aadhaar)
    ).fetchone()

    conn.close()

    if user:
        session["user"] = email
        return redirect(url_for("plastic"))

    return "Invalid login details"


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- PLASTIC FORM ----------------
@app.route("/plastic")
def plastic():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("plastic.html")


# ---------------- PLASTIC SUBMIT ----------------
@app.route("/plastic_submit", methods=["POST"])
def plastic_submit():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    # Generate current date and time
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
    INSERT INTO plastic_data(
        user_email,family_size,house_type,resident_name,
        house_no,apartment_name,flat_no,street,landmark,
        ward_number,local_body,district,pincode,
        milk_packets,deliveries,oil_type,bottles,
        segregation,hks_frequency,receipt,date_time
    )
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,

    (
        session["user"],
        request.form["family_size"],
        request.form["house_type"],
        request.form["resident_name"],
        request.form["house_no"],
        request.form.get("apartment_name"),
        request.form.get("flat_no"),
        request.form["street"],
        request.form.get("landmark"),
        request.form["ward_number"],
        request.form["local_body"],
        request.form["district"],
        request.form["pincode"],
        request.form["milk_packets"],
        request.form["deliveries"],
        request.form["oil_type"],
        request.form["bottles"],
        request.form["segregation"],
        request.form["hks_frequency"],
        request.form["receipt"],
        date_time
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("biodegradable"))


# ---------------- BIO FORM ----------------
@app.route("/biodegradable")
def biodegradable():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("biodegradable.html")


# ---------------- BIO SUBMIT ----------------
@app.route("/biodegradable_submit", methods=["POST"])
def biodegradable_submit():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    # Generate current date and time
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
    INSERT INTO biodegradable_data
    (user_email,food_waste,compost,garden_waste,date_time)

    VALUES (?,?,?,?,?)
    """,

    (
        session["user"],
        request.form["food_waste"],
        request.form["compost"],
        request.form["garden_waste"],
        date_time
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("ewaste"))

# ---------------- EWASTE PAGE ----------------
@app.route("/ewaste")
def ewaste():

    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("ewaste.html")


# ---------------- EWASTE SUBMIT ----------------
@app.route("/ewaste_submit", methods=["POST"])
def ewaste_submit():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    # Generate current date and time
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn.execute("""
    INSERT INTO ewaste_data
    (user_email,old_devices,batteries,disposal_method,date_time)

    VALUES (?,?,?,?,?)
    """,

    (
        session["user"],
        request.form["old_devices"],
        request.form["batteries"],
        request.form["disposal_method"],
        date_time
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("success"))


# ---------------- SUCCESS PAGE ----------------
@app.route("/success")
def success():

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template("success.html",time=timestamp)


# ---------------- RUN APP ----------------
if __name__ == "__main__":

    init_db()

    app.run(debug=True)