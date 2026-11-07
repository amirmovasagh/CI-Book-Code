# Example 3.6: The effect of different convolution filters on a real image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import data, color
 
img = color.rgb2gray(data.retina())[300:1100, 300:1100]     # fundus (retina) image
kernels = {
    "Blur (mean 5x5)": np.ones((5, 5)) / 25,
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Edge (Sobel-x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Edge (Laplacian)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
}
fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original")
for a, (name, k) in zip(ax[1:], kernels.items()):
    out = convolve(img, k)
    shown = np.abs(out) if "Edge" in name else out
    a.imshow(shown, cmap="gray", vmax=np.percentile(shown, 99.5)); a.set_title(name)
    print(f"{name:<18} kernel {k.shape} -> output range [{out.min():.2f}, {out.max():.2f}]")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig3_2.png", dpi=300)
