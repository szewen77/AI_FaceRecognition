# Create a quick check script - save as check_users.py
import sqlite3

conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

print("=== ENROLLED USERS ===")
cursor.execute("SELECT * FROM enrolled_users")
users = cursor.fetchall()
for user in users:
    print(f"ID: {user[0]}, Name: {user[1]}, Date: {user[2]}, Samples: {user[3]}")

print(f"\nTotal enrolled users: {len(users)}")
conn.close()