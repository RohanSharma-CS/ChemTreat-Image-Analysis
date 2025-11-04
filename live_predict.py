import os
import cv2
import numpy as np
import tensorflow as tf
from model import build_model

# -----------------------------------------------------------------------------
# 1️⃣ Load your TensorFlow model structure
# -----------------------------------------------------------------------------
print("Initializing model architecture...")
model = build_model()
print("✅ Model structure loaded successfully.")

# If you have trained weights, uncomment and update this path:
# model.load_weights("/Users/rohans/Desktop/Capstone/ChemTreat-Image-Analysis/turbidity_model.h5")
# print("✅ Trained weights loaded successfully.")

# -----------------------------------------------------------------------------
# 2️⃣ Path to your videos
# -----------------------------------------------------------------------------
video_folder = os.path.expanduser("~/Desktop/Capstone videos")

# -----------------------------------------------------------------------------
# 3️⃣ Define static crop coordinates for the four jars (x, y, width, height)
# -----------------------------------------------------------------------------
# These are tuned for your 854×480 video. Adjust slightly if needed.
jars = [
    (25, 90, 290, 340),    # Jar 1 - moved down, slightly shorter
    (235, 90, 190, 340),   # Jar 2 - same offset as Jar 1
    (440, 90, 180, 340),   # Jar 3 - narrowed on right side
    (635, 90, 180, 340)    # Jar 4 - shifted left, same height
]


# -----------------------------------------------------------------------------
# 4️⃣ Process each video
# -----------------------------------------------------------------------------
for file_name in os.listdir(video_folder):
    if not file_name.lower().endswith(".mov"):
        continue

    video_path = os.path.join(video_folder, file_name)
    print(f"\n🎥 Processing video: {file_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open {file_name}")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        overlay_frame = frame.copy()

        # ---------------------------------------------------------------------
        # Predict turbidity for each jar
        # ---------------------------------------------------------------------
        for i, (x, y, w, h) in enumerate(jars):
            jar_crop = frame[y:y+h, x:x+w]
            jar_resized = cv2.resize(jar_crop, (224, 224))
            jar_input = np.expand_dims(jar_resized / 255.0, axis=(0,1))  # shape (1,1,224,224,3)

            prediction = model.predict(jar_input, verbose=0)
            turbidity = float(prediction[0][0])

            # Print in terminal
            print(f"Frame {frame_count} | Jar {i+1}: {turbidity:.2f}")

            # Overlay text on video
            cv2.putText(overlay_frame, f"Jar {i+1}: {turbidity:.1f}",
                        (x+30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

            # Draw rectangle around jar
            cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display overlay (press 'q' to quit)
        cv2.imshow("Multi-Jar Turbidity", overlay_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    print(f"✅ Finished {file_name} ({frame_count} frames processed)")

cv2.destroyAllWindows()
print("\n🎉 All videos processed successfully!")
