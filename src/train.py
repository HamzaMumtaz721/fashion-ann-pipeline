import os
import yaml
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

with open("params.yaml") as f:
    cfg = yaml.safe_load(f)["train"]

d = np.load("data/processed/data.npz")

model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(cfg["dense_units"], activation="relu"),
    layers.Dropout(cfg["dropout_rate"]),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    d["x_train"],
    d["y_train"],
    validation_data=(d["x_val"], d["y_val"]),
    epochs=cfg["epochs"],
    batch_size=cfg["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)