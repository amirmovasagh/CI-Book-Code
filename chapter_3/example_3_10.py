# Example 3.10: Simulating X-ray imaging with the Beer-Lambert law
# I = I0 * exp(-∫ μ dx)  — each tissue absorbs X-rays differently
import numpy as np
import matplotlib.pyplot as plt
 
# a simple 3D "phantom": soft-tissue body, two lungs (air) and a spine (bone)
n = 160
z, y, x = np.mgrid[-1:1:n*1j, -1:1:n*1j, -1:1:n*1j]
mu = np.zeros((n, n, n))
mu[(x**2 / 0.8 + y**2 / 0.5) < 1] = 0.20                       # soft tissue
for cx in (-0.35, 0.35):
    mu[((x - cx)**2 / 0.06 + y**2 / 0.12 + z**2 / 0.5) < 1] = 0.02  # lungs (air)
mu[((x)**2 + (y + 0.45)**2) < 0.012] = 0.50                    # spine (bone)
mu[((x - 0.1)**2 + (y + 0.05)**2 + (z - 0.2)**2) < 0.006] = 0.35  # a nodule
 
I0, dx = 1.0, 2.0 / n * 10
projection = I0 * np.exp(-mu.sum(axis=1) * dx)   # X-rays travel along the y axis
 
print("Attenuation coefficients used (soft, air, bone):", 0.20, 0.02, 0.50)
print("Detector image shape:", projection.shape)
print("Transmitted intensity range: [%.3f, %.3f]" % (projection.min(), projection.max()))
 
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
ax[0].imshow(mu[:, n // 2, :].T, cmap="gray", origin="lower")
ax[0].set_title("Coronal slice of the phantom (μ map)")
ax[1].imshow(1 - projection, cmap="gray")   # radiograph convention: dense = bright
ax[1].set_title("Simulated radiograph")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig("fig3_5.png", dpi=300)
