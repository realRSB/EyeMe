import time
from collections import deque

class BlinkDetector:
    def __init__(self, eye_closed_thresh=0.30, consec_frames=3):
        self.eye_closed_thresh = eye_closed_thresh
        self.consec_frames = consec_frames
        self.closed_frames = 0
        self.blink_counter = 0
        self.blink_log = deque(maxlen=300)  # Timestamps of detected blinks

    def euclidean(self, p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    def calculate_ear(self, landmarks, eye_indices):
        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        top = (self.euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]]) +
               self.euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]]))
        hor = self.euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        return (top / 2) / hor if hor != 0 else 0

    def update(self, avg_ear):
        if avg_ear < self.eye_closed_thresh:
            self.closed_frames += 1
        else:
            if self.closed_frames >= self.consec_frames:
                self.blink_counter += 1
                self.blink_log.append(time.time())
            self.closed_frames = 0

    def get_blink_rate(self, window=60):
        # Returns the number of blinks detected within the past 'window' seconds
        current_time = time.time()
        return len([t for t in self.blink_log if current_time - t <= window])
