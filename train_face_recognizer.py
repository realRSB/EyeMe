import cv2
import os
import numpy as np
import json

# Paths
DATASET_DIR = "face_training"
MODEL_PATH = "models/face_model.xml"
LABEL_MAP_PATH = "models/label_map.json"

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare training data
faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    label_map[current_label] = person

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        try:
            face_resized = cv2.resize(img, (200, 200))
            faces.append(face_resized)
            labels.append(current_label)
        except Exception as e:
            print(f"Skipping image {img_file} due to error: {e}")

    current_label += 1

if len(faces) == 0:
    print("❌ No training data found.")
else:
    recognizer.train(faces, np.array(labels))
    os.makedirs("models", exist_ok=True)
    recognizer.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)

    print(f"✅ Model trained and saved to {MODEL_PATH}")
    print(f"🗂️ Label map saved to {LABEL_MAP_PATH}")
