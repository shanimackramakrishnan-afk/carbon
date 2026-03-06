import pandas as pd
import numpy as np

# Load existing 500 dataset
existing_data = pd.read_csv("datasets/500_household_dataset_cleaned.csv")

# Number of new samples
n_samples = 1500

# Generate Gaussian distributed values
family_size = np.round(np.random.normal(loc=4, scale=1.2, size=n_samples))
milk_packets = np.round(np.random.normal(loc=3, scale=1.5, size=n_samples))
deliveries = np.round(np.random.normal(loc=15, scale=8, size=n_samples))
old_devices = np.round(np.random.normal(loc=1, scale=1, size=n_samples))
batteries = np.round(np.random.normal(loc=8, scale=5, size=n_samples))

# Clip values to realistic ranges
family_size = np.clip(family_size,1,6)
milk_packets = np.clip(milk_packets,0,6)
deliveries = np.clip(deliveries,0,30)
old_devices = np.clip(old_devices,0,5)
batteries = np.clip(batteries,0,20)

# Food waste categories
food_options = ["less_250g","250g_500g","500g_1kg","more_1kg"]
food_waste = np.random.choice(food_options,n_samples)

# Create dataframe
new_data = pd.DataFrame({
    "family_size":family_size.astype(int),
    "milk_packets":milk_packets.astype(int),
    "deliveries":deliveries.astype(int),
    "food_waste":food_waste,
    "old_devices":old_devices.astype(int),
    "batteries":batteries.astype(int)
})

# Save generated dataset
new_data.to_csv("datasets/1500_gaussian_household_data.csv",index=False)

print("1500 Gaussian datasets generated")