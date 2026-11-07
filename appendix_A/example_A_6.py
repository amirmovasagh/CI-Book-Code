# Example A.6: Working with tabular data in Pandas
import pandas as pd
from sklearn.datasets import load_breast_cancer
 
data = load_breast_cancer(as_frame=True)
df = data.frame
df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})
 
print("Shape:", df.shape)
print(df[["mean radius", "mean texture", "diagnosis"]].head())
print("\nCounts per class:\n", df["diagnosis"].value_counts())
print("\nMean radius per class:\n", df.groupby("diagnosis")["mean radius"].mean().round(2))
print("\nMissing values in the whole table:", int(df.isna().sum().sum()))
large = df[df["mean radius"] > 20]
print("Tumors with mean radius > 20:", len(large), "| malignant share:",
      round((large["diagnosis"] == "malignant").mean(), 3))
