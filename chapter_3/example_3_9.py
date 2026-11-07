# Example 3.9: Anatomical slices — axial, coronal, and sagittal from a 3D volume
import numpy as np
import matplotlib.pyplot as plt
 
n = 128
z, y, x = np.mgrid[-1:1:n*1j, -1:1:n*1j, -1:1:n*1j]
vol = np.zeros((n, n, n))
vol[(x**2 / 0.55 + y**2 / 0.75 + z**2 / 0.9) < 1] = 0.6        # "head"
vol[(x**2 / 0.45 + y**2 / 0.65 + z**2 / 0.8) < 1] = 0.3        # "brain"
vol[((x - 0.25)**2 + (y - 0.2)**2 + (z - 0.15)**2) < 0.02] = 1.0  # a lesion
 
print("Volume shape (z, y, x):", vol.shape)
print("Number of axial slices:", vol.shape[0])
slices = {"Axial (transverse)": vol[n // 2 + 10, :, :],
          "Coronal": vol[:, n // 2 + 13, :],
          "Sagittal": vol[:, :, n // 2 + 16]}
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a, (name, s) in zip(ax, slices.items()):
    a.imshow(s, cmap="gray", origin="lower"); a.set_title(name); a.axis("off")
plt.tight_layout(); plt.savefig("fig3_7.png", dpi=300)
