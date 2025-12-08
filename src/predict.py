import sys
import numpy as np
import tensorflow as tf

from .config import MODEL_PATH
from .utils_extract import extract_settling_features

def predict_turbidity(video_path: str):
    """
    Given a path to a video, returns predicted turbidity per jar.
    """
    feats = extract_settling_features(video_path)   # (num_jars, num_features)
    x = np.expand_dims(feats, axis=0)               # -> (1, num_jars, num_features)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mse')

    preds = model.predict(x)[0]                     # -> (num_jars,)
    return preds

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    preds = predict_turbidity(video_path)

    print(f"\nVIDEO: {video_path}")
    for i, val in enumerate(preds):
        print(f"  Jar {i}: predicted turbidity ≈ {val:.2f}")
