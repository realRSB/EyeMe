import numpy as np
import cv2

def calc_redness(frame, eye_pts):
    """
    Create a mask over the eye region defined by eye_pts and then count
    the fraction of red-dominant pixels as a proxy for redness.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for pt in eye_pts:
        cv2.circle(mask, pt, 2, 255, -1)
    eye_area = cv2.bitwise_and(frame, frame, mask=mask)
    r, g, b = cv2.split(eye_area)
    red_pixels = np.sum(r > 150)
    total_pixels = np.sum(mask > 0) + 1  # Avoid division by zero
    red_ratio = red_pixels / total_pixels
    return red_ratio
