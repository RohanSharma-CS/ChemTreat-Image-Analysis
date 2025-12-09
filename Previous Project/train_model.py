import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_model

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
video_folder = os.path.expanduser("~/Desktop/Capstone videos")
output_model_path = "turbidity_model.keras"
n_frames = 1     # start simple: 1 frame per clip
height, width = 224, 224

# ---------------------------------------------------------------------
# 1️⃣ Collect video paths
# ---------------------------------------------------------------------
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)
               if f.lower().endswith(".mov") or f.lower().endswith(".mp4")]

if not video_files:
    raise ValueError("No video files found in your Capstone videos folder.")

print(f"Found {len(video_files)} videos for training.")

# ---------------------------------------------------------------------
# 2️⃣ Assign dummy turbidity labels (replace these later with real values)
# ---------------------------------------------------------------------
# For now, assign random labels between 10 and 100
# Later, replace this with your actual turbidity readings per video.
labels = np.random.uniform(10, 100, size=len(video_files))

# ---------------------------------------------------------------------
# 3️⃣ Extract frames from each video
# ---------------------------------------------------------------------
def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Couldn't read {video_path}")
    frame = cv2.resize(frame, (width, height))
    return frame / 255.0

X = np.array([extract_first_frame(v) for v in video_files])
y = np.array(labels)

print(f"X shape: {X.shape}, y shape: {y.shape}")

# Add frame dimension for model input: (samples, n_frames, height, width, 3)
X = np.expand_dims(X, axis=1)

# ---------------------------------------------------------------------
# 4️⃣ Split into train/validation sets
# ---------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------------
# 5️⃣ Build model
# ---------------------------------------------------------------------
model = build_model(n_frames=n_frames, height=height, width=width)

# ---------------------------------------------------------------------
# 6️⃣ Train model
# ---------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=4
)

# ---------------------------------------------------------------------
# 7️⃣ Save trained model weights
# ---------------------------------------------------------------------
model.save(output_model_path)
print(f"✅ Model saved successfully as {output_model_path}")
