from ml.waste_estimator import detect_fraud, clean_input, estimate_monthly_waste

data = {
    "family_size": 3,
    "milk_packets": 3,
    "deliveries": 10,
    "food_waste": "250g_500g",
    "old_devices": 1,
    "batteries": 2
}

fraud = detect_fraud(data)

data = clean_input(data)

waste = estimate_monthly_waste(data)

print("Fraud Score:", fraud)
print("Estimated Monthly Waste:", waste, "kg")