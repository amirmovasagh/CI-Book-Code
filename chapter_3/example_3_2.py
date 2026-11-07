# Example 3.2: Automatic cell counting with a classic image processing pipeline
# Gray -> Edge -> Dilate -> Fill holes -> Remove small objects -> Label
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, color, filters, morphology, measure
 
rgb = data.immunohistochemistry()
# Color deconvolution: separate the hematoxylin stain (cell nuclei)
hed = color.rgb2hed(rgb)
nuclei = hed[:, :, 0]
 
smooth = filters.gaussian(nuclei, sigma=1.5)                # 0) denoise
edges = filters.sobel(smooth)                               # 1) edge image
mask = smooth > np.percentile(smooth, 82)                   # 2) keep darkest-stained pixels
mask = morphology.binary_opening(mask, morphology.disk(2))  # 3) erosion + dilation
mask = ndi.binary_fill_holes(mask)                          # 4) fill holes
mask = morphology.remove_small_objects(mask, max_size=60)   # 5) remove small objects
labels = measure.label(mask)                                # 6) label connected parts
props = measure.regionprops(labels)
 
areas = [p.area for p in props]
print("Number of detected nuclei:", labels.max())
print("Mean nucleus area (pixels):", round(np.mean(areas), 1))
print("Largest / smallest area:", max(areas), "/", min(areas))
 
fig, ax = plt.subplots(1, 4, figsize=(15, 4))
ax[0].imshow(rgb); ax[0].set_title("Original (IHC)")
ax[1].imshow(edges, cmap="magma"); ax[1].set_title("Edges")
ax[2].imshow(mask, cmap="gray"); ax[2].set_title("Binary mask")
ax[3].imshow(rgb); ax[3].contour(mask, [0.5], colors="lime", linewidths=0.6)
ax[3].set_title(f"{labels.max()} nuclei detected")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig3_3.png", dpi=300)
