# Example 3.3: Extracting HOG features (histogram of oriented gradients)
import matplotlib.pyplot as plt
from skimage import data, color, exposure
from skimage.feature import hog
from skimage.transform import resize
 
img = resize(color.rgb2gray(data.astronaut()), (256, 256))
features, hog_img = hog(img, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True)
print("Image size:", img.shape)
print("HOG feature vector length:", features.shape[0])
 
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Input")
ax[1].imshow(exposure.rescale_intensity(hog_img, in_range=(0, 0.08)), cmap="gray")
ax[1].set_title("HOG visualization")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig3_4.png", dpi=300)
