import cv2
import numpy as np
from .config import WIDTH, HEIGHT, CROP_COORDS

def load_crop_coords():
    with open(CROP_COORDS, "r") as f:
        x, y, w, h = map(int, f.read().strip().split(","))
    return x, y, w, h

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read {video_path}")
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    return frame.astype("float32") / 255.0

def crop_frame(frame):
    x, y, w, h = load_crop_coords()
    return frame[y:y+h, x:x+w]
