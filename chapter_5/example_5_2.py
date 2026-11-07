# Example 5.2: The KNN algorithm from scratch — exactly the four steps on the slide
import numpy as np
from collections import Counter
 
def knn_predict(X_train, y_train, x_new, k=3):
    # 1) distance from the new point to every training point (Euclidean)
    d = np.sqrt(((X_train - x_new) ** 2).sum(axis=1))
    # 2) sort the data by distance (ascending)
    order = np.argsort(d)
    # 3) choose the first k points as neighbors
    neighbors = order[:k]
    # 4) majority vote of the neighbors' labels
    vote = Counter(y_train[neighbors]).most_common(1)[0][0]
    return vote, d[neighbors]
 
# toy data: two features (e.g. hemoglobin, white-cell count), label 0 = healthy, 1 = anemic
X_train = np.array([[13.5, 7.0], [14.2, 6.5], [15.0, 8.1], [11.0, 6.9],
                    [10.2, 7.5], [9.8, 6.0], [12.9, 7.2], [10.9, 8.0]])
y_train = np.array([0, 0, 0, 1, 1, 1, 0, 1])
x_new = np.array([12.3, 7.6])
 
for k in (1, 3, 5):
    label, dist = knn_predict(X_train, y_train, x_new, k)
    print(f"k={k}: neighbor distances={dist.round(2)} -> predicted class {label}")
