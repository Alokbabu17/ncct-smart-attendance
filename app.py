from flask import Flask, request, jsonify, render_template, redirect, url_for
import psycopg2
import json
import numpy as np
from deepface import DeepFace
import os

app = Flask(__name__)

# --- Cloud Database Configuration ---
# Cloud environment variable se uthayega, nahi milne par aapki Neon connection string use karega
DEFAULT_DB_URL = "postgresql://neondb_owner:npg_8lUcsfNxu9gC@ep-red-sound-b3oxoiux-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

UPLOAD_FOLDER = os.path.join("static", "uploads")
RECEIVED_FOLDER = "received_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECEIVED_FOLDER, exist_ok=True)

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def cosine_distance(source_rep, test_rep):
    a = np.array(source_rep)
    b = np.array(test_rep)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1 - (dot_product / (norm_a * norm_b))

# ----------------- WEB DASHBOARD ROUTES -----------------

@app.route("/", methods=["GET"])
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch logs from Cloud DB
    cursor.execute("""
        SELECT id, trainee_id, trainee_name, course, timestamp, center, status 
        FROM attendance_logs 
        ORDER BY id DESC
    """)
    logs = cursor.fetchall()
    
    # Fetch trainees from Cloud DB
    cursor.execute("""
        SELECT id, trainee_id, name, course, center, photo_path 
        FROM trainees 
        ORDER BY id DESC
    """)
    trainees = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return render_template("index.html", logs=logs, trainees=trainees)

@app.route("/register", methods=["POST"])
def register_trainee():
    trainee_id = request.form.get("trainee_id")
    name = request.form.get("name")
    course = request.form.get("course")
    center = request.form.get("center")
    photo_file = request.files.get("photo")

    if not photo_file or not trainee_id or not name:
        return "Missing details", 400

    save_path = os.path.join(UPLOAD_FOLDER, f"{trainee_id}_{photo_file.filename}")
    photo_file.save(save_path)

    try:
        # Extract facial embedding with face crop
        embedding_objs = DeepFace.represent(
            img_path=save_path,
            model_name="VGG-Face",
            detector_backend="opencv",
            enforce_detection=True
        )
        face_vector = embedding_objs[0]["embedding"]
        embedding_str = json.dumps(face_vector)

        # Save to Cloud PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trainees (trainee_id, name, course, center, photo_path, face_embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (trainee_id) 
            DO UPDATE SET 
                name = EXCLUDED.name,
                course = EXCLUDED.course,
                center = EXCLUDED.center,
                photo_path = EXCLUDED.photo_path,
                face_embedding = EXCLUDED.face_embedding;
        """, (trainee_id, name, course, center, save_path, embedding_str))
        
        conn.commit()
        cursor.close()
        conn.close()

        print(f"[✔] Cloud DB Registered: {name} ({trainee_id})")
        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Face Registration Failed: Ensure clear face in photo. Error: {str(e)}", 400

# ----------------- HARDWARE API ROUTE -----------------

@app.route("/api/v1/attendance/verify", methods=["POST"])
def verify_attendance():
    print("\n==========================================")
    print("🔔 [EVENT] Terminal Attendance Check (Cloud DB)")
    print("==========================================")

    if "image" not in request.files:
        return jsonify({"status": "failed", "matched": False, "message": "No image sent"}), 400

    file = request.files["image"]
    file_path = os.path.join(RECEIVED_FOLDER, "temp_capture.jpg")
    file.save(file_path)

    try:
        try:
            incoming_rep = DeepFace.represent(
                img_path=file_path,
                model_name="VGG-Face",
                detector_backend="opencv",
                enforce_detection=True
            )
        except Exception:
            incoming_rep = DeepFace.represent(
                img_path=file_path,
                model_name="VGG-Face",
                detector_backend="skip"
            )

        if not incoming_rep:
            return jsonify({"status": "failed", "matched": False, "message": "Face extraction failed"}), 200

        target_embedding = incoming_rep[0]["embedding"]

        # Fetch embeddings from Cloud PostgreSQL
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT trainee_id, name, course, center, face_embedding FROM trainees")
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            print("[-] WARNING: Cloud Database me koi registered student nahi mila!")
            return jsonify({"status": "failed", "matched": False, "message": "No registered trainees"}), 200

        matched_trainee = None
        min_distance = 1.0
        THRESHOLD = 0.58

        for row in rows:
            t_id, name, course, center, emb_str = row
            saved_embedding = json.loads(emb_str)

            dist = cosine_distance(target_embedding, saved_embedding)
            print(f"    --> Comparing with {name} | Distance: {dist:.4f}")

            if dist < min_distance:
                min_distance = dist
                if dist <= THRESHOLD:
                    matched_trainee = {
                        "trainee_id": t_id,
                        "name": name,
                        "course": course,
                        "center": center
                    }

        if matched_trainee:
            # Insert attendance log into Cloud DB
            cursor.execute("""
                INSERT INTO attendance_logs (trainee_id, trainee_name, course, center, status)
                VALUES (%s, %s, %s, %s, 'PRESENT')
            """, (
                matched_trainee["trainee_id"], 
                matched_trainee["name"], 
                matched_trainee["course"], 
                matched_trainee["center"]
            ))
            conn.commit()
            cursor.close()
            conn.close()

            print(f"[✔] MATCH SUCCESS: {matched_trainee['name']} Logged into Cloud DB!")
            return jsonify({
                "status": "success",
                "matched": True,
                "trainee": matched_trainee
            }), 200
        else:
            cursor.close()
            conn.close()
            print(f"[X] NO MATCH (Best: {min_distance:.4f}, Threshold: {THRESHOLD})")
            return jsonify({
                "status": "failed",
                "matched": False,
                "message": "Face not recognized"
            }), 200

    except Exception as e:
        print(f"[-] EXCEPTION: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Render ya local host dono par automatically correct port bind karega
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] NCCT Server running on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)