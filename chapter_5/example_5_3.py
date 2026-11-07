# Example 5.3: The KNN decision boundary for k=1 and k=15 (comparison with the slide)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
 
X, y = make_blobs(n_samples=300, centers=[(-3, 3), (0, -2), (3, 3)], cluster_std=2.0, random_state=4)
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
light = ListedColormap(["#f7b6b2", "#b3c6f7", "#b8e6b0"]); dark = ["#d62728", "#1f3fbf", "#2ca02c"]
 
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, k in zip(ax, [1, 15]):
    m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, cmap=light)
    for c in range(3):
        a.scatter(*X[y == c].T, s=12, c=dark[c])
    a.set_title(f"K = {k}  (training accuracy = {m.score(X, y):.2f})")
    print(f"k={k:>2}: training accuracy = {m.score(X, y):.3f}")
plt.tight_layout(); plt.savefig("fig5_2.png", dpi=300)
