# Example 1.7: Data preprocessing — missing values, categorical variables, and scaling
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
 
df = pd.DataFrame({
    "age":        [45, 61, np.nan, 38, 72, 55],
    "glucose":    [98, 180, 130, np.nan, 210, 115],
    "sex":        ["F", "M", "M", "F", "F", "M"],
    "smoker":     ["no", "yes", "no", "no", "yes", np.nan],
})
print("Raw data:\n", df, "\n")
print("Missing values per column:\n", df.isna().sum().to_dict(), "\n")
 
numeric, categorical = ["age", "glucose"], ["sex", "smoker"]
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder()), categorical),
])
X = prep.fit_transform(df)
print("Output columns:", list(prep.get_feature_names_out()))
print("Processed matrix (rounded):\n", np.round(X, 2))
 
scores = np.array([[1.0], [5.0], [9.0]])
print("\nStandardScaler:", StandardScaler().fit_transform(scores).ravel().round(3))
print("MinMaxScaler  :", MinMaxScaler().fit_transform(scores).ravel())
