import cv2
import numpy as np
import os
from src.config import RAW_VIDEO_DIR, JAR_COORDS

def extract_curve(video_path):
    cap = cv2.VideoCapture(video_path)
    curves = [[] for _ in range(4)]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i, (x,y,w,h) in enumerate(JAR_COORDS):
            crop = gray[y:y+h, x:x+w]
            mean_val = float(np.mean(crop))
            curves[i].append(mean_val)

    cap.release()
    return curves

def save_curves(video_name, curves):
    out_dir = "data/curves"
    os.makedirs(out_dir, exist_ok=True)

    for i, curve in enumerate(curves):
        out_path = os.path.join(out_dir, f"{video_name}_jar{i}.csv")
        np.savetxt(out_path, curve, delimiter=",")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    for fname in os.listdir(RAW_VIDEO_DIR):
        if not fname.endswith(".mp4"):
            continue
        
        full_path = os.path.join(RAW_VIDEO_DIR, fname)
        print(f"\nExtracting curves from {fname} ...")

        curves = extract_curve(full_path)
        save_curves(fname.replace(".mp4",""), curves)
