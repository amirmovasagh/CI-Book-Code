# Example 3.11: CT scanning — from projection images to a reconstructed slice
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale
 
phantom = rescale(shepp_logan_phantom(), 0.5)       # 200x200 head phantom
for n_angles in (18, 180):
    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(phantom, theta=theta)               # projections from many angles
    recon = iradon(sino, theta=theta, filter_name="ramp")
    err = np.sqrt(np.mean((recon - phantom) ** 2))
    print(f"{n_angles:>3} angles -> sinogram {sino.shape}, reconstruction RMSE = {err:.4f}")
 
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(phantom, cmap="gray"); ax[0].set_title("Original slice")
ax[1].imshow(sino, cmap="gray", aspect="auto"); ax[1].set_title("Sinogram (180 angles)")
ax[1].set_xlabel("angle"); ax[1].set_ylabel("detector position")
ax[2].imshow(recon, cmap="gray"); ax[2].set_title("Filtered back-projection")
ax[0].axis("off"); ax[2].axis("off")
plt.tight_layout(); plt.savefig("fig3_6.png", dpi=300)
