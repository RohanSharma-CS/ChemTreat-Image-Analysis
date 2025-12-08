import os
import sys
import numpy as np
import pandas as pd
import joblib

from .config import JAR_COORDS, CURVE_MODEL_PATH
from .utils_extract import extract_settling_curve
from .build_features import extract_features


def predict_video(video_path):
    """Extract curves → compute features → predict turbidity."""

    print(f"\nVIDEO: {video_path}")

    # Load model
    model = joblib.load(CURVE_MODEL_PATH)

    # Columns model was trained with
    feature_names = [
        "mean", "std", "min", "max",
        "drop", "slope_full", "slope_init", "roughness"
    ]

    results = {}

    # Loop through jars
    for j, box in enumerate(JAR_COORDS):

        curve = extract_settling_curve(video_path, box)

        if len(curve) < 5:
            print(f"  Jar {j}: ❌ Not enough frames to extract features")
            continue

        feat_vector = extract_features(curve)   # 1D → 8 features
        X_df = pd.DataFrame([feat_vector], columns=feature_names)

        pred = model.predict(X_df)[0]
        results[j] = pred

        print(f"  Jar {j}: predicted turbidity ≈ {pred:.2f}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_curve <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print("❌ Video not found:", video_path)
        sys.exit(1)

    predict_video(video_path)


if __name__ == "__main__":
    main()
