import os
import pickle
import pandas as pd
from src.build_features import extract_features
from src.extract_curves import extract_curves_from_video
from src.config import CURVE_DIR

def predict(video_path):
    # extract curves first
    extract_curves_from_video(video_path)

    # load model
    with open("models/turbidity_regressor.pkl", "rb") as f:
        model = pickle.load(f)

    results = {}

    base = os.path.splitext(os.path.basename(video_path))[0]

    for jar in range(4):
        curve_path = os.path.join(CURVE_DIR, f"{base}_jar{jar}.csv")
        df = pd.read_csv(curve_path)

        feats = extract_features(df["brightness"])
        X = pd.DataFrame([feats])

        turb = model.predict(X)[0]
        results[f"jar{jar}"] = float(turb)

    return results

if __name__ == "__main__":
    import sys
    vid = sys.argv[1]
    print(predict(vid))

