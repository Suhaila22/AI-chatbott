import sqlite3

DB_NAME = "enterprise_chatbot.db"

username = input("bbbb: ").strip()

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

cursor.execute("UPDATE users SET role = 'admin' WHERE username = ?", (username,))
conn.commit()

if cursor.rowcount == 0:
    print("No user found with this username. Register the user first.")
else:
    print(f"User '{username}' is now admin. Logout and login again.")

conn.close()
