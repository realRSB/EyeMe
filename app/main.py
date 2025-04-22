import cv2
import time
import os
from datetime import datetime
import mediapipe as mp

from vision.blink_detection import BlinkDetector
from vision.redness_detection import calc_redness
from vision.pupil_dilation import calculate_pupil_diameter
from vision.health_score import compute_eye_health_score

from recognizer.eye_identifier import identify_eye_snapshot
from utils.logging_helpers import log_data, create_timestamped_log_file, ensure_logs_folder
from utils.snapshot_helpers import crop_and_save_eye_snapshot

# ========== Setup ==========
ensure_logs_folder()
log_filename = create_timestamped_log_file()
today = datetime.now().strftime("%Y-%m-%d")
eye_image_saved = False
os.makedirs("data/snapshots", exist_ok=True)

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Facial landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]

# Trackers
blink_detector = BlinkDetector(eye_closed_thresh=0.30)
timestamps, blink_rates = [], []
start_time, last_log_time = time.time(), 0

# ========== Utility ==========
def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def draw_eye_outline(frame, landmarks, eye_indices, img_w, img_h, color=(0, 255, 255)):
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], color, 1)

# ========== Main Loop ==========
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = blink_detector.calculate_ear(landmarks, LEFT_EYE)
        right_ear = blink_detector.calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        blink_detector.update(avg_ear)
        current_blink_rate = blink_detector.get_blink_rate(window=60)

        elapsed_min = (time.time() - start_time) / 60
        timestamps.append(elapsed_min)
        blink_rates.append(current_blink_rate)

        left_eye_pts = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in LEFT_EYE]
        redness = calc_redness(frame, left_eye_pts)
        redness_label = "HIGH" if redness > 0.05 else "NORMAL"
        pupil_diameter = calculate_pupil_diameter(landmarks, LEFT_IRIS, euclidean)
        health_score, strain_level = compute_eye_health_score(current_blink_rate, redness, blink_detector.blink_log, time.time() - start_time)

        if not eye_image_saved:
            eye_landmarks = LEFT_EYE + RIGHT_EYE
            temp_path = crop_and_save_eye_snapshot(frame, landmarks, eye_landmarks, "data/snapshots", filename_prefix="temp_")

            name = identify_eye_snapshot(temp_path, threshold=70)
            cv2.putText(frame, f"User: {name}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            final_path = f"data/snapshots/{name}_{today}.jpg"
            os.rename(temp_path, final_path)
            eye_image_saved = True

        if time.time() - last_log_time >= 5:
            log_data(log_filename, {
                "time_min": elapsed_min,
                "blink_rate": current_blink_rate,
                "redness": redness,
                "pupil_diameter": pupil_diameter,
                "health_score": health_score,
                "strain_level": strain_level
            })
            last_log_time = time.time()

        for idx in LEFT_EYE + RIGHT_EYE + LEFT_IRIS:
            pt = landmarks[idx]
            x, y = int(pt.x * img_w), int(pt.y * img_h)
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        draw_eye_outline(frame, landmarks, LEFT_EYE, img_w, img_h)
        draw_eye_outline(frame, landmarks, RIGHT_EYE, img_w, img_h)

        # Overlay text
        cv2.putText(frame, f"Blinks: {blink_detector.blink_counter}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink Rate: {current_blink_rate:.1f}/min", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Redness: {redness_label}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if redness_label == "HIGH" else (0, 255, 255), 2)
        cv2.putText(frame, f"Pupil Diam.: {pupil_diameter:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Strain: {strain_level}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        cv2.putText(frame, f"Eye Health: {health_score}/100", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    cv2.imshow("Smart Eye Health Tracker", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
