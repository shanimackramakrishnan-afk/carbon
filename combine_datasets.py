import pandas as pd

data1 = pd.read_csv("datasets/500_household_dataset_cleaned.csv")
data2 = pd.read_csv("datasets/1500_gaussian_household_data.csv")


combined = pd.concat([data1,data2],ignore_index=True)

print("Total datasets:",len(combined))

combined.to_csv("datasets/2000_household_waste_dataset.csv",index=False)

print("Final dataset saved")
