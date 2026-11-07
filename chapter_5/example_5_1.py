# Example 5.1: Eager versus lazy learning — comparing training and prediction times
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 
X, y = make_classification(n_samples=40000, n_features=20, random_state=0)
X_new = X[:5000]
 
for name, model in [("KNN (lazy)", KNeighborsClassifier(5, algorithm="brute")),
                    ("Decision tree (eager)", DecisionTreeClassifier(random_state=0))]:
    t0 = time.perf_counter(); model.fit(X, y);       t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); model.predict(X_new);  t_pred = time.perf_counter() - t0
    print(f"{name:<22} fit: {t_fit*1000:8.1f} ms | predict 5000 samples: {t_pred*1000:8.1f} ms")
