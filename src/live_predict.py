import os
import cv2
import numpy as np
import tensorflow as tf
from .config import RAW_VIDEO_DIR, MODEL_PATH

# ---------------------------------------------------------
# 4 JAR COORDINATES (x, y, w, h)
# ---------------------------------------------------------
jars = [
    (25, 90, 190, 340),    # Jar 1
    (235, 90, 190, 340),   # Jar 2
    (440, 90, 180, 340),   # Jar 3
    (635, 90, 180, 340)    # Jar 4
]

# ---------------------------------------------------------
# Helper: Crop a jar from the full frame
# ---------------------------------------------------------
def crop_jar(frame, coords):
    x, y, w, h = coords
    return frame[y:y+h, x:x+w]


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------
def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    for fname in os.listdir(RAW_VIDEO_DIR):
        if not fname.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        path = os.path.join(RAW_VIDEO_DIR, fname)
        print(f"Processing video: {fname}")

        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            overlay = frame.copy()

            # Process each jar
            for i, coords in enumerate(jars):
                jar_crop = crop_jar(frame, coords)

                if jar_crop.size == 0:
                    print(f"⚠️ Empty crop for Jar {i+1}, skipping.")
                    continue

                jar_resized = cv2.resize(jar_crop, (224, 224))
                jar_input = np.expand_dims(jar_resized / 255.0, axis=(0, 1))

                prediction = model.predict(jar_input, verbose=0)
                turbidity = float(prediction[0][0])

                # Draw bounding box & label
                x, y, w, h = coords
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"Jar {i+1}: {turbidity:.1f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("4-Jar Turbidity Predictions", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
