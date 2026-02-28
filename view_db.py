import sqlite3
from tabulate import tabulate

conn = sqlite3.connect("database/waste.db")
cursor = conn.cursor()


# USERS TABLE
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()

print("\nUSERS TABLE")
print(tabulate(users,
               headers=["ID","Name","Email","Password","Aadhaar"],
               tablefmt="grid"))


# PLASTIC TABLE
cursor.execute("SELECT * FROM plastic_data")
plastic = cursor.fetchall()

print("\nPLASTIC DATA TABLE")
print(tabulate(plastic,
               headers=["ID","User Email","Plastic Bags","Bottled Water","Packaging"],
               tablefmt="grid"))


# BIODEGRADABLE TABLE
cursor.execute("SELECT * FROM biodegradable_data")
bio = cursor.fetchall()

if bio:
    print("\nBIODEGRADABLE TABLE")
    print(tabulate(bio,
                   headers=["ID","User Email","Food Waste","Compost","Garden Waste"],
                   tablefmt="grid"))


# EWASTE TABLE
cursor.execute("SELECT * FROM ewaste_data")
ewaste = cursor.fetchall()

if ewaste:
    print("\nEWASTE TABLE")
    print(tabulate(ewaste,
                   headers=["ID","User Email","Old Devices","Batteries","Disposal"],
                   tablefmt="grid"))

conn.close()