import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import threading
from datetime import datetime
import os

from modules.blink_detection import BlinkDetector
from modules.redness_detection import calc_redness
from modules.pupil_dilation import calculate_pupil_diameter
from modules.health_score import compute_eye_health_score
from modules.data_logger import log_data

os.makedirs("snapshots", exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
eye_snapshot_path = f"snapshots/eye_{today}.jpg"
eye_image_saved = False


# Initialize Mediapipe Face Mesh with refined landmarks (for iris data)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Define indices for blink detection and iris:
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]

# Global variables for plotting/logging and blink detection.
timestamps = []
blink_rates = []
blink_detector = BlinkDetector(eye_closed_thresh=0.30)
start_time = time.time()
last_log_time = 0  # Track the last time we logged data

eye_image_saved = False
os.makedirs("snapshots", exist_ok=True)

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def draw_eye_outline(frame, landmarks, eye_indices, img_w, img_h, color=(0, 255, 255)):
    points = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i+1) % len(points)], color, 1)

def plot_blink_graph():
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        if timestamps:
            ax.clear()
            ax.plot(timestamps, blink_rates, label="Blink Rate (blinks/min)")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Blinks per Minute")
            ax.set_ylim(0, 30)
            ax.set_title("Blink Rate Over Time")
            ax.legend()
            plt.pause(5)

threading.Thread(target=plot_blink_graph, daemon=True).start()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR for both eyes using the BlinkDetector module.
        left_ear = blink_detector.calculate_ear(landmarks, LEFT_EYE)
        right_ear = blink_detector.calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        blink_detector.update(avg_ear)
        current_blink_rate = blink_detector.get_blink_rate(window=60)

        elapsed_time = time.time() - start_time  # in seconds
        elapsed_min = elapsed_time / 60
        timestamps.append(elapsed_min)
        blink_rates.append(current_blink_rate)

        # Redness detection using landmarks from the left eye.
        left_eye_pts = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in LEFT_EYE]
        redness = calc_redness(frame, left_eye_pts)
        redness_label = "HIGH" if redness > 0.05 else "NORMAL"

        # Pupil (iris) diameter estimation using refined landmarks.
        pupil_diameter = calculate_pupil_diameter(landmarks, LEFT_IRIS, euclidean)

        # Compute overall eye health score and strain level; pass elapsed_time for warm-up check.
        health_score, strain_level = compute_eye_health_score(current_blink_rate, redness, blink_detector.blink_log, elapsed_time)

        if not eye_image_saved:
            eye_landmarks = LEFT_EYE + RIGHT_EYE
            xs = [int(landmarks[i].x * img_w) for i in eye_landmarks]
            ys = [int(landmarks[i].y * img_h) for i in eye_landmarks]
            x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, img_w)
            y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, img_h)

            eye_crop = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(eye_snapshot_path, eye_crop)
            eye_image_saved = True

        current_time = time.time()
        if current_time - last_log_time >= 5:  # Log every 5 seconds
            data = {
                "time_min": elapsed_min,
                "blink_rate": current_blink_rate,
                "redness": redness,
                "pupil_diameter": pupil_diameter,
                "health_score": health_score,
                "strain_level": strain_level
            }
            log_data("eye_health_log.csv", data)
            last_log_time = current_time

        # Visualize landmarks.
        for idx in LEFT_EYE:
            pt = landmarks[idx]
            x, y = int(pt.x * img_w), int(pt.y * img_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for idx in RIGHT_EYE:
            pt = landmarks[idx]
            x, y = int(pt.x * img_w), int(pt.y * img_h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Draw iris landmarks.
        for idx in LEFT_IRIS:
            pt = landmarks[idx]
            x, y = int(pt.x * img_w), int(pt.y * img_h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # Draw outlines for the eyes.
        draw_eye_outline(frame, landmarks, LEFT_EYE, img_w, img_h)
        draw_eye_outline(frame, landmarks, RIGHT_EYE, img_w, img_h)

        # Overlay text with metrics.
        cv2.putText(frame, f"Blinks: {blink_detector.blink_counter}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink Rate: {current_blink_rate:.1f}/min", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Redness: {redness_label}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if redness_label=="HIGH" else (0, 255, 255), 2)
        cv2.putText(frame, f"Pupil Diam.: {pupil_diameter:.2f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Strain: {strain_level}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        cv2.putText(frame, f"Eye Health: {health_score}/100", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

    cv2.imshow("Smart Eye Health Tracker", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit.
        break

cap.release()
cv2.destroyAllWindows()
