import pandas as pd

print("Loading dataset...")

df = pd.read_csv("datasets/FINAL_ML_READY_DATASET.csv")

print("Dataset loaded")
print("Rows:", len(df))


# ---------------- WASTE ESTIMATION ----------------

def estimate_waste(row):

    # ---------------- PLASTIC WASTE ----------------

    milk_plastic = row["milk_packets"] * 5 * 30
    delivery_plastic = row["deliveries"] * 50

    plastic_kg = (milk_plastic + delivery_plastic) / 1000

    # Plastic risk multiplier (plastic is highly harmful)
    plastic_weighted = plastic_kg * 2.2


    # ---------------- BIO WASTE ----------------

    food_map = {
        "less_250g": 5,
        "250g_500g": 10,
        "500g_1kg": 20,
        "more_1kg": 30
    }

    bio_kg = food_map.get(row["food_waste"], 10)

    # Bio waste multiplier (normal)
    bio_weighted = bio_kg * 1.0


    # ---------------- E-WASTE ----------------

    battery_map = {
        "none": 0,
        "1-2": 0.1,
        "3-5": 0.25,
        "6+": 0.5
    }

    ewaste_kg = (row["old_devices"] * 1.5) + battery_map.get(row["batteries"], 0)

    # E-waste toxicity multiplier
    ewaste_weighted = ewaste_kg * 1.3


    # ---------------- TOTAL WASTE ----------------

    total = plastic_weighted + bio_weighted + ewaste_weighted


    return pd.Series([
        plastic_kg,
        bio_kg,
        ewaste_kg,
        total
    ])


print("Estimating monthly waste...")

df[
    [
        "plastic_waste_kg",
        "bio_waste_kg",
        "ewaste_kg",
        "monthly_waste_kg"
    ]
] = df.apply(estimate_waste, axis=1)

print("Waste columns created")


# Save dataset
df.to_csv("datasets/2000_household_waste_dataset.csv", index=False)

print("Saved dataset with waste columns")

pd.set_option('display.max_columns', None)

print(df.head())