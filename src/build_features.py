import os
import pandas as pd
import numpy as np

from .config import CURVE_DIR, LABELS_CURVE_CSV, FEATURE_CSV


def extract_features(curve):
    """Convert a settling curve (brightness over time) into numeric features."""
    curve = np.array(curve).astype(float)

    # Handle empty curve
    if len(curve) < 2:
        return [0, 0, 0, 0, 0, 0, 0, 0]

    mean = float(np.mean(curve))
    std = float(np.std(curve))
    min_val = float(np.min(curve))
    max_val = float(np.max(curve))
    drop = float(curve[-1] - curve[0])

    # slope across full curve
    slope_full = float((curve[-1] - curve[0]) / len(curve))

    # slope across first 10%
    k = max(1, len(curve) // 10)
    slope_init = float((curve[k] - curve[0]) / k)

    # roughness = average |Δ|
    diffs = np.abs(np.diff(curve))
    roughness = float(np.mean(diffs))

    # RETURN FLAT NUMERIC LIST (critical!)
    return [
        mean, std, min_val, max_val,
        drop, slope_full, slope_init, roughness
    ]


def main():
    print("[INFO] Loading labels:", LABELS_CURVE_CSV)
    df_labels = pd.read_csv(LABELS_CURVE_CSV)

    rows = []

    for _, row in df_labels.iterrows():
        video = row["video"]
        jar = int(row["jar"])
        turbidity = float(row["turbidity"])

        curve_file = os.path.join(CURVE_DIR, f"{video}_jar{jar}.csv")

        if not os.path.exists(curve_file):
            print(f"[WARNING] Missing curve file: {curve_file}")
            continue

        # Load curve
        curve = pd.read_csv(curve_file, header=None)[0].values

        # Extract feature vector
        feats = extract_features(curve)

        rows.append([*feats, turbidity, video, jar])

    # Build DataFrame
    cols = [
        "mean", "std", "min", "max",
        "drop", "slope_full", "slope_init", "roughness",
        "turbidity", "video", "jar"
    ]

    df_out = pd.DataFrame(rows, columns=cols)

    print("[INFO] Saving features to:", FEATURE_CSV)
    df_out.to_csv(FEATURE_CSV, index=False)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
