import sqlite3

conn = sqlite3.connect("database/user.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(biodegradable_data)")
columns = cursor.fetchall()

for col in columns:
    print(col)

conn.close()