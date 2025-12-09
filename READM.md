# ChemTreat Prototype — Turbidity Estimation From Video

## 📌 Project Overview
This project builds a prototype ML pipeline that estimates turbidity from jar-test settling videos.
It extracts brightness curves from each jar, converts them into numeric features, and uses a regression model to predict turbidity values.

The current version supports multiple jars per video, automatic curve extraction, feature engineering, plotting, and turbidity prediction.

## :busts_in_silhouette: Members
* Aubrey Carey
* Rohan Sharma
* Virginia Anderson
* Ronald Menendez

## 🎯 Objectives
- Replace traditional manual settling tests with a vision-based system.
- Objectively measure settling rates and turbidity reduction.
- Enable real-time process monitoring and optimization.
- Provide consistent and reproducible chemical performance comparisons.

## :file_folder: Project Structure
```
ChemTreat-Prototype/
│
├── data/
│   ├── raw_videos/          # Input videos
│   ├── curves/              # Extracted brightness curves per jar
│   ├── features.csv         # Engineered features for ML
│   ├── labels_curve.csv     # True turbidity labels
│
├── models/
│   ├── turbidity_model.keras   # (Old CNN, not used now)
│   ├── curve_model.pkl         # RandomForest turbidity predictor
│
├── src/
│   ├── config.py               # Paths, jar coordinates, model paths
│   ├── extract_curves.py       # Extract brightness curves from videos
│   ├── build_features.py       # Convert curves → feature vectors
│   ├── train_curve_model.py    # Train RandomForest model
│   ├── predict_curve.py        # Predict turbidity for a new video
│   ├── plot_curves.py          # Plot brightness curves visually
│   ├── utils_extract.py        # Core curve extraction logic
│
└── README.md
```

# :pushpin: What the System Does
### :one: Extract Brightness Curves from Each Jar

For every video, the system crops each jar using the coordinates in config.py:

```
JAR_COORDS = [
    (25, 90, 190, 340),
    (235, 90, 190, 340),
    (440, 90, 180, 340),
    (635, 90, 180, 340)
]
```


Each jar produces a 1-D brightness-over-time signal (its settling curve).

### :two: Build Feature Vectors

From each curve, the system computes features such as:

- mean brightness

- standard deviation

- min / max

- initial-to-final brightness drop

- slope

- roughness

This becomes a single row in features.csv.

### :three: Train ML Model

A RandomForestRegressor is trained using:

- Input: extracted features

- Target: true turbidity (from labels_curve.csv)

The model is saved to:

- `models/curve_model.pkl`

### :four: Predict Turbidity for New Videos

Given a new video, the system:

- extracts brightness curves

- builds features

- predicts turbidity for each jar

Example output:

```python
VIDEO: data/raw_videos/IMG_7810.mp4
  Jar 0: predicted turbidity ≈ 17.51
  Jar 1: predicted turbidity ≈ 17.27
  Jar 2: predicted turbidity ≈ 23.53
  Jar 3: predicted turbidity ≈ 25.90
  ```

## :rocket: How to Run the Pipeline
1. Add videos

* Place .mp4 files in:

  * `data/raw_videos/`

2. Add turbidity labels

* Edit:

  * `data/labels_curve.csv`


Example:

```
video3,0,20
video3,1,28
video3,2,8
video3,3,65
```
3. Extract brightness curves

```powershell 
python -m src.extract_curves
```


This writes CSVs into `data/curves/`.

4. Build features

```powershell
python -m src.build_features
```

This creates:

`data/features.csv`

5. Train the turbidity model

```powershell
python -m src.train_curve_model
```


This produces:

* `models/curve_model.pkl`

6. Predict turbidity for a new video

```powershell
python -m src.predict_curve data/raw_videos/IMG_7810.mp4
```

7. Plot settling curves (optional)

```powershell
python -m src.plot_curves data/raw_videos/IMG_7810.mp4
```

## :chart_with_upwards_trend: Current Limitations

Only ~8 total samples → model accuracy is limited.

Lighting/lens placement affects brightness curves.

More labeled videos are required for meaningful prediction accuracy.

## :crystal_ball: Next Steps

Collect 5–10 more labeled videos.

Retrain model with more samples.

Add smoothing filters to reduce vibration noise.

Build a user interface for real-time turbidity display.

Support automatic jar detection instead of fixed coordinates.