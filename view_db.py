import sqlite3
import pandas as pd
from tabulate import tabulate

# connect database
conn = sqlite3.connect("database/user.db")
cursor = conn.cursor()

print("="*80)
print("FULL DATABASE VIEW".center(80))
print("="*80)

# get all tables in database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

for table in tables:
    table_name = table[0]

    print("\nTable:", table_name)
    print("-"*80)

    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

        if not df.empty:
            print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
            print("Total Records:", len(df))
        else:
            print("Table is empty")

    except Exception as e:
        print("Error reading table:", e)

print("\n" + "="*80)
print("END DATABASE VIEW".center(80))
print("="*80)

conn.close()