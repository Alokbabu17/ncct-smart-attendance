import os
import sys
import cv2
import numpy as np
import psycopg2
import json
from flask import Flask, request, jsonify, render_template, redirect, url_for

app = Flask(__name__)

DEFAULT_DB_URL = "postgresql://neondb_owner:npg_8lUcsfNxu9gC@ep-red-sound-b3oxoiux-pooler.c-4.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

UPLOAD_FOLDER = os.path.join("static", "uploads")
RECEIVED_FOLDER = "received_frames"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECEIVED_FOLDER, exist_ok=True)

# Load SFace & YuNet Models
SFACE_PATH = os.path.join(os.path.dirname(__file__), "models", "sface.onnx")
YUNET_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet.onnx")

recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), score_threshold=0.3, nms_threshold=0.3)

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[-] Error: Image could not be loaded from {image_path}", flush=True)
        return None

    h, w, _ = img.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        print(f"[-] Detection Warning: No face detected in frame", flush=True)
        return None

    aligned_face = recognizer.alignCrop(img, faces[0])
    feature = recognizer.feature(aligned_face)
    return feature.flatten().tolist()

def match_cosine(a, b):
    # Pure NumPy implementation: Immune to C++ OpenCV shape assertion crashes
    vec_a = np.array(a, dtype=np.float32).flatten()
    vec_b = np.array(b, dtype=np.float32).flatten()
    
    if vec_a.shape != vec_b.shape:
        return -1.0  # Dimension mismatch (Old model embedding)

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

# --- KEEP-ALIVE ROUTE ---
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive", "server": "NCCT Production API"}), 200

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
            return "Registration Failed: Clear face not detected. Please upload a clear frontal photo.", 400

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

        print(f"[✔] Registered: {name} ({trainee_id}) [Vector len: {len(face_vector)}]", flush=True)
        return redirect(url_for("dashboard"))

    except Exception as e:
        print(f"[-] Registration Error: {str(e)}", flush=True)
        return f"Error: {str(e)}", 400

# ----------------- HARDWARE API ROUTE -----------------

@app.route("/api/v1/attendance/verify", methods=["POST"])
def verify_attendance():
    print("\n==========================================", flush=True)
    print("🔔 [EVENT] New Attendance Scan Received!", flush=True)
    print("==========================================", flush=True)

    if "image" not in request.files:
        print("[-] Rejected: No image sent", flush=True)
        return jsonify({"status": "failed", "matched": False, "message": "No image sent"}), 400

    file = request.files["image"]
    file_path = os.path.join(RECEIVED_FOLDER, "temp_capture.jpg")
    file.save(file_path)

    try:
        target_embedding = extract_embedding(file_path)
        if target_embedding is None:
            print("[-] Verification Failed: Face not detected in camera frame", flush=True)
            return jsonify({"status": "failed", "matched": False, "message": "Face not detected clearly"}), 200

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT trainee_id, name, course, center, face_embedding FROM trainees")
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            print("[-] Warning: No registered trainees in DB", flush=True)
            return jsonify({"status": "failed", "matched": False, "message": "No registered trainees"}), 200

        matched_trainee = None
        best_score = -1.0
        THRESHOLD = 0.30  # Practical cosine similarity for SFace

        for row in rows:
            t_id, name, course, center, emb_str = row
            saved_embedding = json.loads(emb_str)

            # Check for legacy vector mismatch
            if len(saved_embedding) != len(target_embedding):
                print(f"    --> [!] Skipping {name}: Outdated embedding size ({len(saved_embedding)} vs {len(target_embedding)}). Needs re-registration!", flush=True)
                continue

            score = match_cosine(target_embedding, saved_embedding)
            print(f"    --> Candidate: {name} | Cosine Match Score: {score:.4f} (Threshold: {THRESHOLD})", flush=True)

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

            print(f"[✔] MATCH SUCCESS: {matched_trainee['name']} marked PRESENT (Score: {best_score:.4f})", flush=True)
            return jsonify({
                "status": "success",
                "matched": True,
                "trainee": matched_trainee
            }), 200
        else:
            cursor.close()
            conn.close()
            print(f"[-] REJECTED: Best score {best_score:.4f} < {THRESHOLD}", flush=True)
            return jsonify({
                "status": "failed",
                "matched": False,
                "message": "Face not recognized"
            }), 200

    except Exception as e:
        print(f"[-] Fatal Exception: {str(e)}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)