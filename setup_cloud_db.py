import psycopg2

# Yahan apni Neon.tech wali exact connection string paste karein
DATABASE_URL = "postgresql://neondb_owner:npg_8lUcsfNxu9gC@ep-red-sound-b3oxoiux-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

def init_cloud_db():
    print("[*] Connecting to Cloud PostgreSQL...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # 1. Trainees Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trainees (
            id SERIAL PRIMARY KEY,
            trainee_id VARCHAR(50) UNIQUE NOT NULL,
            name VARCHAR(100) NOT NULL,
            course VARCHAR(150) NOT NULL,
            center VARCHAR(100) NOT NULL,
            photo_path TEXT,
            face_embedding TEXT NOT NULL
        );
        """)

        # 2. Attendance Records Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_logs (
            id SERIAL PRIMARY KEY,
            trainee_id VARCHAR(50) NOT NULL,
            trainee_name VARCHAR(100) NOT NULL,
            course VARCHAR(150) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            center VARCHAR(100) NOT NULL,
            status VARCHAR(20) NOT NULL,
            CONSTRAINT fk_trainee FOREIGN KEY (trainee_id) REFERENCES trainees(trainee_id)
        );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("[✔] Cloud PostgreSQL Tables ('trainees', 'attendance_logs') created successfully!")

    except Exception as e:
        print(f"[-] Connection Failed: {str(e)}")

if __name__ == "__main__":
    init_cloud_db()