import tensorflow as tf
from tensorflow.keras import layers, models

"""
model.py
---------
This file defines and returns the turbidity prediction model architecture.

It is a cleaned version of the original ChemTreat model.
All Windows-based file paths, dataset loading, and training code
have been removed so it can be imported safely by live_predict.py.
"""

def build_model(n_frames=1, height=224, width=224):
    """
    Builds and compiles the 3D CNN model for turbidity prediction.
    You can later load pretrained weights using model.load_weights(path).
    """
    model = models.Sequential([
        # Input layer for video sequences
        layers.Input(shape=(n_frames, height, width, 3)),

        # 3D convolutional layers
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((1, 2, 2)),

        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((1, 2, 2)),

        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),

        # Flatten and dense layers
        layers.Reshape((n_frames * 56 * 56 * 64,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='softplus')  # Turbidity output
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )

    return model
