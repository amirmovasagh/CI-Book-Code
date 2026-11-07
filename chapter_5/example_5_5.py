# Example 5.5: The K-Means algorithm from scratch, showing its steps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
 
X, _ = make_blobs(n_samples=240, centers=3, cluster_std=1.1, random_state=2)
rng = np.random.default_rng(5)
k = 3
centers = X[rng.choice(len(X), k, replace=False)]        # 1) random initial centers
 
snapshots = []
for it in range(10):
    d = np.linalg.norm(X[:, None, :] - centers[None], axis=2)
    labels = d.argmin(1)                                 # 2) assign each point to nearest center
    snapshots.append((centers.copy(), labels.copy()))
    new_centers = np.array([X[labels == c].mean(0) for c in range(k)])   # 3) move to centroid
    sse = (d.min(1) ** 2).sum()
    print(f"iteration {it + 1}: SSE = {sse:8.2f}")
    if np.allclose(new_centers, centers):                # 5) stop when nothing changes
        print("Converged.")
        break
    centers = new_centers                                # 4) repeat
 
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, i in zip(ax, [0, 1, len(snapshots) - 1]):
    c, l = snapshots[i]
    a.scatter(*X.T, c=l, cmap="viridis", s=12); a.scatter(*c.T, c="red", marker="X", s=220, edgecolor="k")
    a.set_title(f"iteration {i + 1}")
plt.tight_layout(); plt.savefig("fig5_3.png", dpi=300)
