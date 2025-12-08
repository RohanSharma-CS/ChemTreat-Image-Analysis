import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

from .config import FEATURE_CSV

MODEL_OUT = "models/curve_model.pkl"

def main():
    print("[INFO] Loading features:", FEATURE_CSV)
    df = pd.read_csv(FEATURE_CSV)

    # Ensure columns exist
    required_cols = {"video", "jar", "turbidity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")

    # -------------------------
    # 1. Separate features / labels
    # -------------------------
    X = df.drop(columns=["turbidity", "video", "jar"])
    y = df["turbidity"]

    print("[INFO] Training regression model on", len(df), "samples")

    # -------------------------
    # 2. Train/test split
    # -------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # -------------------------
    # 3. RandomForestRegressor (simple + robust)
    # -------------------------
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("[INFO] Training complete.")
    print("[INFO] Validation R²:", model.score(X_val, y_val))

    # -------------------------
    # 4. Save model
    # -------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    print("[INFO] Saved curve model to:", MODEL_OUT)


if __name__ == "__main__":
    main()
