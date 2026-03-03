import sqlite3
import csv
import pandas as pd
from tabulate import tabulate

conn = sqlite3.connect("database/waste.db")

# Export function
def export_to_csv(table_name, filename):
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_csv(f"{filename}.csv", index=False)
    print(f"✅ Exported {table_name} to {filename}.csv")

# Display function
def display_table(table_name, title):
    print(f"\n{title}")
    print("-"*80)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    if not df.empty:
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
        print(f"📊 Total Records: {len(df)}")
        return df
    else:
        print(f"No data in {table_name}")
        return None

# Display all tables
print("="*80)
print("📊 WASTE MANAGEMENT DATABASE REPORT".center(80))
print("="*80)

# Users table
users_df = display_table("users", "👤 USERS TABLE")
if users_df is not None:
    export_to_csv("users", "users_export")

# Plastic table
plastic_df = display_table("plastic_data", "🥤 PLASTIC WASTE DATA TABLE")
if plastic_df is not None:
    export_to_csv("plastic_data", "plastic_waste_export")

# Biodegradable table
bio_df = display_table("biodegradable_data", "🌱 BIODEGRADABLE WASTE DATA TABLE")
if bio_df is not None:
    export_to_csv("biodegradable_data", "biodegradable_waste_export")

# E-waste table
ewaste_df = display_table("ewaste_data", "📱 E-WASTE DATA TABLE")
if ewaste_df is not None:
    export_to_csv("ewaste_data", "ewaste_export")

print("\n" + "="*80)
print("🏁 END OF REPORT".center(80))
print("="*80)

conn.close()