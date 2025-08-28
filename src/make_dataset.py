# src/make_dataset.py
from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

data = load_iris(as_frame=True)
df = data.frame
df["species"] = df["target"].map({i: name for i, name in enumerate(data.target_names)})
df = df.drop(columns=["target"])
df.to_csv("data/iris.csv", index=False)
print("Saved dataset to data/iris.csv")
