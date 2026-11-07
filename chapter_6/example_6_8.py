# Example 6.8: Fuzzy C-Means clustering — each point belongs to all clusters to some degree
import numpy as np
from sklearn.datasets import make_blobs
 
X, _ = make_blobs(n_samples=150, centers=[(0, 0), (4, 4)], cluster_std=1.2, random_state=3)
c, m, rng = 2, 2.0, np.random.default_rng(0)
U = rng.dirichlet(np.ones(c), size=len(X))          # random initial memberships (rows sum to 1)
for it in range(100):
    Um = U ** m
    centers = (Um.T @ X) / Um.sum(0)[:, None]
    d = np.linalg.norm(X[:, None] - centers[None], axis=2) + 1e-9
    U_new = 1 / ((d[:, :, None] / d[:, None, :]) ** (2 / (m - 1))).sum(2)
    if np.abs(U_new - U).max() < 1e-5: break
    U = U_new
print("Converged after", it + 1, "iterations")
print("Centers:\n", centers.round(2))
mid = np.argmin(np.abs(U[:, 0] - 0.5))
print("A point between the two clusters:", X[mid].round(2), "-> memberships", U[mid].round(2))
print("A typical point:", X[0].round(2), "-> memberships", U[0].round(2))
