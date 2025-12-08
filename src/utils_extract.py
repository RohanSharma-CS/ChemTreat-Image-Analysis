import cv2
import numpy as np
from .config import JAR_COORDS

def extract_settling_curves(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video: {video_path}")

    curves = [[] for _ in JAR_COORDS]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for j, (x, y, w, h) in enumerate(JAR_COORDS):
            crop = gray[y:y+h, x:x+w]
            curves[j].append(float(np.mean(crop)))

    cap.release()
    return [np.array(c) for c in curves]
