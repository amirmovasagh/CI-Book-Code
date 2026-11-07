# Example 4.8: Interpretability with an occlusion sensitivity map
import numpy as np
import matplotlib.pyplot as plt
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import example_4_2 as cnn        # re-uses (and re-trains) the CNN of Example 4.2 (saved as example_4_2.py)
 
def occlusion_map(img, label, size=3):
    base = cnn.forward(img[None])[-1][0, label]
    heat = np.zeros_like(img)
    for i in range(8 - size + 1):
        for j in range(8 - size + 1):
            x = img.copy(); x[i:i + size, j:j + size] = 0          # hide a 3x3 patch
            drop = base - cnn.forward(x[None])[-1][0, label]
            heat[i:i + size, j:j + size] += drop
    return base, heat
 
fig, ax = plt.subplots(2, 4, figsize=(11, 5.4))
for k, idx in enumerate([0, 3, 7, 11]):
    img, label = cnn.X_te[idx], cnn.y_te[idx]
    base, heat = occlusion_map(img, label)
    print(f"test image {idx}: digit {label}, P(correct class) = {base:.3f}, "
          f"most important region (row, col) = {tuple(int(v) for v in np.unravel_index(heat.argmax(), heat.shape))}")
    ax[0, k].imshow(img, cmap="gray"); ax[0, k].set_title(f"digit {label}")
    ax[1, k].imshow(img, cmap="gray"); ax[1, k].imshow(heat, cmap="jet", alpha=0.55)
    ax[1, k].set_title("occlusion sensitivity")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig("fig4_4.png", dpi=300)
