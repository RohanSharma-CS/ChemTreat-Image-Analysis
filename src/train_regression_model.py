import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

def main():
    df = pd.read_csv("data/features.csv")

    X = df.drop(columns=["turbidity", "video", "jar"])
    y = df["turbidity"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    model.fit(X, y)

    with open("models/turbidity_regressor.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved turbidity regressor → models/turbidity_regressor.pkl")

if __name__ == "__main__":
    main()
