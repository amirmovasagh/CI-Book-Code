# Example 4.5: Data augmentation for medical images
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import data, color, transform
 
rng = np.random.default_rng(0)
img = transform.resize(color.rgb2gray(data.retina()), (256, 256))
 
aug = {
    "Original": img,
    "Horizontal flip": img[:, ::-1],
    "Rotation 15°": ndi.rotate(img, 15, reshape=False, mode="reflect"),
    "Zoom 1.2x": transform.resize(img[21:235, 21:235], (256, 256)),
    "Brightness +20%": np.clip(img * 1.2, 0, 1),
    "Gaussian noise": np.clip(img + rng.normal(0, 0.04, img.shape), 0, 1),
}
for name, a in aug.items():
    print(f"{name:<16} shape={a.shape} mean intensity={a.mean():.3f}")
 
fig, ax = plt.subplots(1, 6, figsize=(16, 3))
for a, (name, im) in zip(ax, aug.items()):
    a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(name, fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig("fig4_3.png", dpi=300)
