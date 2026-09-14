import sqlite3
import json

def init_db():
    conn = sqlite3.connect("ncct.db")
    cursor = conn.cursor()

    # 1. Trainees Table (Profile + Face Representation)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trainees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trainee_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        course TEXT NOT NULL,
        center TEXT NOT NULL,
        photo_path TEXT,
        face_embedding TEXT
    )
    """)

    # 2. Attendance Records Table (Logs marked by ESP32)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trainee_id TEXT NOT NULL,
        trainee_name TEXT NOT NULL,
        course TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        center TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (trainee_id) REFERENCES trainees(trainee_id)
    )
    """)

    conn.commit()
    conn.close()
    print(" Database 'ncct.db' and tables initialized successfully.")

if __name__ == "__main__":
    init_db()