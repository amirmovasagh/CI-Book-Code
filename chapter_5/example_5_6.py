# Example 5.6: Choosing the number of clusters with the elbow method and the silhouette score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=10)
ks, inertia, sil = range(2, 10), [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}: inertia (SSE) = {km.inertia_:8.1f} | silhouette = {sil[-1]:.3f}")
print("Best k by silhouette:", ks[sil.index(max(sil))])
 
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("SSE")
ax[1].plot(ks, sil, "o-", c="#d1495b"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig("fig5_4.png", dpi=300)
