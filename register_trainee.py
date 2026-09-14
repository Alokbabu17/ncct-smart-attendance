import sqlite3
import json
from deepface import DeepFace
import os

def register_new_trainee(trainee_id, name, course, center, image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' nahi mili!")
        return

    print("Extracting facial representation with face crop...")
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_path,
            model_name="VGG-Face",
            detector_backend="opencv",   # Face crop karega
            enforce_detection=True
        )
        
        face_vector = embedding_objs[0]["embedding"]
        embedding_str = json.dumps(face_vector)

        conn = sqlite3.connect("ncct.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trainees 
            (trainee_id, name, course, center, photo_path, face_embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (trainee_id, name, course, center, image_path, embedding_str))

        conn.commit()
        conn.close()
        print(f"Successfully re-registered with face crop: {name} ({trainee_id})")

    except Exception as e:
        print(f"Registration Failed: {str(e)}")

if __name__ == "__main__":
    register_new_trainee(
        trainee_id="NCCT-2026-001",
        name="Alok Raj",
        course="Smart Agro & PACS Management",
        center="RICM Bhopal",
        image_path="my_face.jpg"
    )