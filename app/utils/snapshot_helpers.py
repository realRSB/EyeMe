import os
import cv2
from datetime import datetime

def crop_and_save_eye_snapshot(frame, landmarks, eye_indices, output_dir, filename_prefix="", padding=10):
    """
    Crop the eye region based on landmarks and save it as an image.

    Parameters:
        frame (np.ndarray): The full video frame
        landmarks (list): Mediapipe facial landmarks
        eye_indices (list): Landmark indices for cropping (e.g., LEFT_EYE + RIGHT_EYE)
        output_dir (str): Folder to save image
        filename_prefix (str): Optional prefix (e.g., user name or temp_)
        padding (int): Margin around eyes

    Returns:
        final_path (str): Path to saved image
    """
    h, w = frame.shape[:2]
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    x_min, x_max = max(min(xs) - padding, 0), min(max(xs) + padding, w)
    y_min, y_max = max(min(ys) - padding, 0), min(max(ys) + padding, h)

    eye_crop = frame[y_min:y_max, x_min:x_max]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_path = os.path.join(output_dir, f"{filename_prefix}{timestamp}.jpg")
    cv2.imwrite(final_path, eye_crop)
    return final_path
