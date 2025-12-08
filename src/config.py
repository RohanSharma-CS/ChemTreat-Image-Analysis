import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw_videos")
LABEL_CSV = os.path.join(DATA_DIR, "labels.csv")

# JAR coordinates
JAR_COORDS = [
    (25, 90, 190, 340),
    (235, 90, 190, 340),
    (440, 90, 180, 340),
    (635, 90, 180, 340)
]

# Curve data directory + labels
CURVE_DIR = os.path.join(DATA_DIR, "curves")
LABELS_CURVE_CSV = os.path.join(DATA_DIR, "labels_curve.csv")

# Feature CSV
FEATURE_CSV = os.path.join(DATA_DIR, "features.csv")

# Model output
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "turbidity_model.keras")
CURVE_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "curve_model.pkl"))

HEIGHT = 224
WIDTH = 224
N_FRAMES = 1
