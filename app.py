import os
import cv2
import numpy as np
import psycopg2
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

# --- Cloud Database Configuration ---
DEFAULT_DB_URL = "postgresql://neondb_owner:npg_8lUcsfNxu9gC@ep-red-sound-b3oxoiux-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

UPLOAD_FOLDER = os.path.join("static", "uploads")
RECEIVED_FOLDER = "received_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECEIVED_FOLDER, exist_ok=True)

# Load SFace Model via OpenCV DNN (Takes < 40MB RAM)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sface.onnx")
recognizer = cv2.FaceRecognizerSF.create(MODEL_PATH, "")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) == 0:
        # Fallback to center crop if haar misses
        h, w = img.shape[:2]
        crop_box = np.array([[w * 0.1, h * 0.1, w * 0.8, h * 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        aligned_face = recognizer.alignCrop(img, crop_box)
    else:
        x, y, w, h = faces[0]
        dummy_face = np.array([[x, y, w, h, x, y, x+w, y, x+w/2, y+h/2, x, y+h, x+w, y+h]], dtype=np.float32)
        aligned_face = recognizer.alignCrop(img, dummy_face)

    embedding = recognizer.feature(aligned_face)
    return embedding.flatten().tolist()

def cosine_distance(a, b):
    vec_a = np.array(a, dtype=np.float32)
    vec_b = np.array(b, dtype=np.float32)
    return float(recognizer.match(vec_a, vec_b, cv2.FaceRecognizerSF_FR_COSINE))

# ----------------- WEB DASHBOARD ROUTES -----------------

@app.route("/", methods=["GET"])
def dashboard():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, trainee_id, trainee_name, course, timestamp, center, status 
        FROM attendance_logs 
        ORDER BY id DESC
    """)
    logs = cursor.fetchall()
    
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
        face_vector = extract_embedding(save_path)
        if face_vector is None:
            return "Could not process face image", 400

        embedding_str = json.dumps(face_vector)

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

        print(f"[✔] Registered: {name} ({trainee_id})")
        return redirect(url_for("dashboard"))

    except Exception as e:
        return f"Registration Error: {str(e)}", 400

# ----------------- HARDWARE API ROUTE -----------------

@app.route("/api/v1/attendance/verify", methods=["POST"])
def verify_attendance():
    if "image" not in request.files:
        return jsonify({"status": "failed", "matched": False, "message": "No image sent"}), 400

    file = request.files["image"]
    file_path = os.path.join(RECEIVED_FOLDER, "temp_capture.jpg")
    file.save(file_path)

    try:
        target_embedding = extract_embedding(file_path)
        if target_embedding is None:
            return jsonify({"status": "failed", "matched": False, "message": "Face extraction failed"}), 200

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT trainee_id, name, course, center, face_embedding FROM trainees")
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            return jsonify({"status": "failed", "matched": False, "message": "No registered trainees"}), 200

        matched_trainee = None
        best_score = 0.0
        # OpenCV SFace Cosine: higher score is a better match (Threshold: 0.363)
        THRESHOLD = 0.363

        for row in rows:
            t_id, name, course, center, emb_str = row
            saved_embedding = json.loads(emb_str)

            score = cosine_distance(target_embedding, saved_embedding)
            print(f"    --> Candidate: {name} | Match Score: {score:.4f}")

            if score > best_score:
                best_score = score
                if score >= THRESHOLD:
                    matched_trainee = {
                        "trainee_id": t_id,
                        "name": name,
                        "course": course,
                        "center": center
                    }

        if matched_trainee:
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

            print(f"[✔] MATCH SUCCESS: {matched_trainee['name']}")
            return jsonify({
                "status": "success",
                "matched": True,
                "trainee": matched_trainee
            }), 200
        else:
            cursor.close()
            conn.close()
            return jsonify({
                "status": "failed",
                "matched": False,
                "message": "Face not recognized"
            }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)