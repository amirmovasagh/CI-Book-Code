# Example 1.4: Clustering without labels using K-Means
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
 
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=7)
 
kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)          # labels are created by the algorithm
 
print("Cluster sizes:", [int((labels == k).sum()) for k in range(3)])
print("Cluster centers:\n", kmeans.cluster_centers_.round(2))
 
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].scatter(X[:, 0], X[:, 1], c="gray", s=15)
ax[0].set_title("Raw data (no labels)")
ax[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=15)
ax[1].scatter(*kmeans.cluster_centers_.T, c="red", marker="X", s=200)
ax[1].set_title("After K-Means (k=3)")
plt.tight_layout(); plt.savefig("fig1_3.png", dpi=300)
