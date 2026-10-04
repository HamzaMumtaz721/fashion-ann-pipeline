import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open("params.yaml") as f:
    cfg = yaml.safe_load(f)["preprocess"]

raw = np.load("data/raw/fashion_mnist.npz")
pixels = raw["x_train"].astype("float32") / 255.0
mean = pixels.mean()
std = pixels.std()
x_train = (pixels - mean) / std
x_test = (raw["x_test"].astype("float32") / 255.0 - mean) / std

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train,
    raw["y_train"],
    test_size=cfg["test_size"],
    random_state=cfg["seed"],
    stratify=raw["y_train"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_tr,
    y_train=y_tr,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=raw["y_test"],
)
print("processed data saved")