from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim=12):
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,)),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1)  # output turbidity value
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    return model
