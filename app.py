from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "waste_secret_key"


# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    if not os.path.exists("database"):
        os.makedirs("database")
    conn = sqlite3.connect("database/waste.db")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- HOME / LOGIN PAGE ----------------
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

    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            aadhaar TEXT UNIQUE NOT NULL
        )
    """)

    try:
        conn.execute(
            "INSERT INTO users (name, email, password, aadhaar) VALUES (?, ?, ?, ?)",
            (name, email, password, aadhaar)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return "User already exists"

    conn.close()
    return redirect(url_for("login"))


# ---------------- LOGIN SUBMIT ----------------
@app.route("/login_submit", methods=["POST"])
def login_submit():
    email = request.form["email"]
    password = request.form["password"]
    aadhaar = request.form["aadhaar"]

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email=? AND password=? AND aadhaar=?",
        (email, password, aadhaar)
    ).fetchone()
    conn.close()

    if user:
        session["user"] = user["email"]
        return redirect(url_for("plastic"))
    else:
        return "Invalid Email, Password, or Aadhaar"


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- PLASTIC PAGE ----------------
@app.route("/plastic")
def plastic():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("plastic.html")


# ---------------- PLASTIC SUBMIT ----------------
@app.route("/plastic_submit", methods=["POST"])
def plastic_submit():
    # all fields from plastic.html
    family_size = request.form["family_size"]
    house_type = request.form["house_type"]
    resident_name = request.form["resident_name"]
    house_no = request.form["house_no"]
    apartment_name = request.form.get("apartment_name")
    flat_no = request.form.get("flat_no")
    street = request.form["street"]
    landmark = request.form.get("landmark")
    ward_number = request.form["ward_number"]
    local_body = request.form["local_body"]
    district = request.form["district"]
    pincode = request.form["pincode"]
    milk_packets = request.form["milk_packets"]
    deliveries = request.form["deliveries"]
    oil_type = request.form["oil_type"]
    bottles = request.form["bottles"]
    segregation = request.form["segregation"]
    hks_frequency = request.form["hks_frequency"]
    receipt = request.form["receipt"]

    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plastic_data (
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
            receipt TEXT
        )
    """)

    conn.execute("""
        INSERT INTO plastic_data (
            user_email,
            family_size,
            house_type,
            resident_name,
            house_no,
            apartment_name,
            flat_no,
            street,
            landmark,
            ward_number,
            local_body,
            district,
            pincode,
            milk_packets,
            deliveries,
            oil_type,
            bottles,
            segregation,
            hks_frequency,
            receipt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user"],
        family_size,
        house_type,
        resident_name,
        house_no,
        apartment_name,
        flat_no,
        street,
        landmark,
        ward_number,
        local_body,
        district,
        pincode,
        milk_packets,
        deliveries,
        oil_type,
        bottles,
        segregation,
        hks_frequency,
        receipt
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("biodegradable"))


# ---------------- BIODEGRADABLE PAGE ----------------
@app.route("/biodegradable")
def biodegradable():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("biodegradable.html")


@app.route("/biodegradable_submit", methods=["POST"])
def biodegradable_submit():
    food_waste = request.form["food_waste"]
    compost = request.form["compost"]
    garden_waste = request.form["garden_waste"]

    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS biodegradable_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            food_waste TEXT,
            compost TEXT,
            garden_waste TEXT
        )
    """)
    conn.execute("""
        INSERT INTO biodegradable_data
        (user_email, food_waste, compost, garden_waste)
        VALUES (?, ?, ?, ?)
    """, (session["user"], food_waste, compost, garden_waste))
    conn.commit()
    conn.close()

    return redirect(url_for("ewaste"))


# ---------------- EWASTE PAGE ----------------
@app.route("/ewaste")
def ewaste():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("ewaste.html")


@app.route("/ewaste_submit", methods=["POST"])
def ewaste_submit():
    old_devices = request.form["old_devices"]
    batteries = request.form["batteries"]
    disposal_method = request.form["disposal_method"]

    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ewaste_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            old_devices TEXT,
            batteries TEXT,
            disposal_method TEXT
        )
    """)
    conn.execute("""
        INSERT INTO ewaste_data
        (user_email, old_devices, batteries, disposal_method)
        VALUES (?, ?, ?, ?)
    """, (session["user"], old_devices, batteries, disposal_method))
    conn.commit()
    conn.close()

    return redirect(url_for("success"))


# ---------------- SUCCESS PAGE ----------------
@app.route("/success")
def success():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("success.html")


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)