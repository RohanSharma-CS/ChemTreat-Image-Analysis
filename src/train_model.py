import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_VIDEO_DIR, LABEL_CSV, MODEL_PATH
from .utils_extract import extract_settling_features
from .model import build_model

def main():
    df = pd.read_csv(LABEL_CSV)
    label_map = dict(zip(df["filename"], df["turbidity"]))

    X = []
    y = []

    for fname in os.listdir(RAW_VIDEO_DIR):
        if fname not in label_map:
            continue

        full_path = os.path.join(RAW_VIDEO_DIR, fname)
        features = extract_settling_features(full_path)

        if features is None:
            print(f"[SKIP] Could not process: {fname}")
            continue

        X.append(features)
        y.append(label_map[fname])

    X = np.array(X)
    y = np.array(y, dtype=np.float32)

    print(f"[INFO] Loaded {len(X)} samples")

    # Handle case: only 1 video
    if len(X) > 1:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = X, y

    model = build_model(input_dim=X.shape[1])
    print("[INFO] Training model...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=1
    )

    model.save(MODEL_PATH)
    print("Saved model at:", MODEL_PATH)

if __name__ == "__main__":
    main()
