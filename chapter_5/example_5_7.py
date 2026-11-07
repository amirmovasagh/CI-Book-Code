# Example 5.7: Segmenting a histology image with K-Means on pixel colors
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from sklearn.cluster import KMeans
 
img = data.immunohistochemistry() / 255.0
pixels = img.reshape(-1, 3)
km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(pixels[::20])   # fit on a sample
labels = km.predict(pixels).reshape(img.shape[:2])
segmented = km.cluster_centers_[labels]
 
for c, center in enumerate(km.cluster_centers_):
    print(f"cluster {c}: mean RGB = {(center * 255).astype(int)} | share = {np.mean(labels == c):.1%}")
 
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img); ax[0].set_title("Original")
ax[1].imshow(segmented); ax[1].set_title("K-Means colors (k=3)")
ax[2].imshow(labels, cmap="tab10"); ax[2].set_title("Cluster labels")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig5_5.png", dpi=300)
