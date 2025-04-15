import cv2
import os
import json

# Paths
MODEL_PATH = "models/face_model.xml"
LABEL_MAP_PATH = "models/label_map.json"

# Load trained recognizer and label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists(MODEL_PATH):
    recognizer.read(MODEL_PATH)
else:
    recognizer = None

if os.path.exists(LABEL_MAP_PATH):
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
else:
    label_map = {}

def identify_eye_snapshot(image_path, threshold=70):
    """
    Identify the person from an eye snapshot using LBPH face recognizer.
    Returns the predicted name or "unknown" if confidence is too low.
    """
    if not recognizer or not label_map:
        return "unknown"

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown"

    try:
        resized = cv2.resize(img, (200, 200))
        label, confidence = recognizer.predict(resized)
        if confidence > threshold:
            return "unknown"
        return label_map.get(label, "unknown")
    except Exception:
        return "unknown"
