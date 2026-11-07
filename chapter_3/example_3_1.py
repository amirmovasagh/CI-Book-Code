# Example 3.1: A digital image from a computer's perspective — a matrix of numbers
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
 
rgb = data.immunohistochemistry()          # pathology slide image (IHC staining)
gray = color.rgb2gray(rgb)                 # convert to grayscale, values in [0, 1]
 
print("RGB image shape :", rgb.shape, "| dtype:", rgb.dtype)
print("Gray image shape:", gray.shape)
print("Pixel (100,100) RGB value:", rgb[100, 100])
print("Top-left 4x4 gray block (x255):\n", (gray[:4, :4] * 255).astype(int))
 
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
ax[0].imshow(rgb); ax[0].set_title("RGB")
for i, (ch, cm) in enumerate([("Red", "Reds"), ("Green", "Greens"), ("Blue", "Blues")]):
    ax[i + 1].imshow(rgb[:, :, i], cmap=cm); ax[i + 1].set_title(f"{ch} channel")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig3_1.png", dpi=300)
