import sqlite3
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

DATABASE_PATH = "database/waste.db"

# ---------------- OPTIONS ----------------

districts = [
    "Thiruvananthapuram","Kollam","Pathanamthitta","Alappuzha",
    "Kottayam","Idukki","Ernakulam","Thrissur",
    "Palakkad","Malappuram","Kozhikode",
    "Wayanad","Kannur","Kasaragod"
]

family_sizes = ["1", "2-3", "4-5", "6+"]
house_types = ["Apartment", "Small house", "Large house"]

oil_types = ["Plastic packet", "Plastic bottle", "Loose / refill"]
bottle_options = ["None", "1-3", "4-7", "7+"]

segregation_options = ["Always", "Sometimes", "Never"]

hks_options = ["Every month", "Every 2-3 months", "Rarely", "Never"]

receipt_options = ["Yes", "No"]

food_waste_options = ["less_250g", "250g_500g", "500g_1kg", "more_1kg"]

compost_options = [
    "home_compost",
    "biomethanation",
    "worm_compost",
    "community",
    "collector",
    "mixed"
]

garden_options = ["daily", "weekly", "monthly", "no"]

disposal_methods = ["recycle", "store", "trash", "sell"]

battery_options = ["none", "1-2", "3-5", "6+"]

# ---------------- HELPER FUNCTIONS ----------------

def generate_email(n):
    return f"user{n}@gmail.com"


def weighted_family():
    return random.choices(
        family_sizes,
        weights=[10, 40, 35, 15]
    )[0]


def milk_by_family(size):

    if size == "1":
        return random.randint(0, 1)

    elif size == "2-3":
        return random.randint(1, 3)

    elif size == "4-5":
        return random.randint(2, 4)

    else:
        return random.randint(3, 6)


def food_by_family(size):

    if size == "1":
        return "less_250g"

    elif size == "2-3":
        return "250g_500g"

    elif size == "4-5":
        return "500g_1kg"

    else:
        return "more_1kg"


# ---------------- DATA GENERATION ----------------

def generate_dataset(num_records=2000):

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    for i in range(1, num_records + 1):

        email = generate_email(i)
        family = weighted_family()
        house = random.choice(house_types)
        district = random.choice(districts)

        milk_packets = milk_by_family(family)
        deliveries = random.randint(2, 20)

        # Plastic Table

        cursor.execute("""
            INSERT INTO plastic_data (
                user_email, family_size, house_type,
                resident_name, house_no,
                apartment_name, flat_no,
                street, landmark,
                ward_number, local_body,
                district, pincode,
                milk_packets, deliveries,
                oil_type, bottles,
                segregation, hks_frequency, receipt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (

            email,
            family,
            house,
            f"Resident_{i}",
            f"{random.randint(1,500)}/{random.randint(1,20)}",
            "",
            "",
            "Main Road",
            "Near School",
            f"Ward {random.randint(1,25)}",
            "Municipality",
            district,
            str(random.randint(670000,689999)),
            milk_packets,
            deliveries,
            random.choice(oil_types),
            random.choice(bottle_options),
            random.choice(segregation_options),
            random.choice(hks_options),
            random.choice(receipt_options)

        ))

        # Biodegradable Table

        cursor.execute("""
            INSERT INTO biodegradable_data
            (user_email, food_waste, compost, garden_waste)
            VALUES (?, ?, ?, ?)
        """, (

            email,
            food_by_family(family),
            random.choice(compost_options),
            random.choice(garden_options)

        ))

        # E-Waste Table

        cursor.execute("""
            INSERT INTO ewaste_data
            (user_email, old_devices, batteries, disposal_method)
            VALUES (?, ?, ?, ?)
        """, (

            email,
            random.randint(0,4),
            random.choice(battery_options),
            random.choice(disposal_methods)

        ))

    conn.commit()
    conn.close()

    print(f"{num_records} synthetic records inserted successfully!")


# ---------------- EXPORT DATABASE TO CSV ----------------

def export_to_csv():

    conn = sqlite3.connect(DATABASE_PATH)

    query = """

    SELECT

        p.user_email,
        p.family_size,
        p.house_type,
        p.district,
        p.milk_packets,
        p.deliveries,
        p.oil_type,
        p.bottles,
        p.segregation,

        b.food_waste,
        b.compost,
        b.garden_waste,

        e.old_devices,
        e.batteries,
        e.disposal_method

    FROM plastic_data p

    LEFT JOIN biodegradable_data b
    ON p.user_email = b.user_email

    LEFT JOIN ewaste_data e
    ON p.user_email = e.user_email

    """

    df = pd.read_sql_query(query, conn)

    df.to_csv("2000_household_dataset.csv", index=False)

    conn.close()

    print("CSV Exported Successfully")
    print("Total rows:", len(df))


# ---------------- REMOVE DUPLICATES ----------------

def clean_dataset():

    df = pd.read_csv("2000_household_dataset.csv")

    print("Before Cleaning:", len(df))

    df_cleaned = df.drop_duplicates(subset=["user_email"])

    print("After Cleaning:", len(df_cleaned))

    df_cleaned.to_csv("2000_household_dataset_cleaned.csv", index=False)

    print("Duplicates removed")


# ---------------- PREPARE ML DATASET ----------------

def prepare_ml_dataset():

    df = pd.read_csv("2000_household_dataset_cleaned.csv")

    print("Original Shape:", df.shape)

    # Handle Missing Values

    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in df.select_dtypes(include=['int64','float64']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Encode Categorical Data

    label_encoder = LabelEncoder()

    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    print("Categorical Encoding Done")

    # Normalize Numeric Features

    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Normalization Done")

    # Save Final Dataset

    df.to_csv("FINAL_ML_READY_DATASET.csv", index=False)

    print("Final ML Dataset Saved")

    print("Final Shape:", df.shape)

    print(df.head())


# ---------------- MAIN ----------------

if __name__ == "__main__":

    generate_dataset(2000)

    export_to_csv()

    clean_dataset()

    prepare_ml_dataset()
