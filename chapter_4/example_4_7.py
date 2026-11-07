# Example 4.7: Pixel-by-pixel segmentation (the FCN idea) and the Dice and IoU metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from sklearn.neural_network import MLPClassifier
 
rng = np.random.default_rng(1)
 
def make_image(n=96, n_cells=7):
    """Synthetic microscopy image with round 'cells' and its ground-truth mask."""
    yy, xx = np.mgrid[:n, :n]
    mask = np.zeros((n, n), bool)
    for _ in range(n_cells):
        cy, cx, r = rng.uniform(10, n - 10, 2).tolist() + [rng.uniform(5, 11)]
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    img = 0.35 + 0.35 * mask + ndi.gaussian_filter(rng.normal(0, 0.25, (n, n)), 1)
    img += 0.15 * np.sin(xx / 15)                  # uneven illumination
    return img, mask
 
def pixel_features(img):
    """A small filter bank = what the conv layers of an FCN learn automatically."""
    feats = [img]
    for s in (1, 2, 4):
        feats += [ndi.gaussian_filter(img, s),
                  ndi.gaussian_laplace(img, s),
                  ndi.gaussian_gradient_magnitude(img, s)]
    return np.stack(feats, -1).reshape(-1, len(feats))
 
def dice(a, b):  return 2 * (a & b).sum() / (a.sum() + b.sum())
def iou(a, b):   return (a & b).sum() / (a | b).sum()
 
train = [make_image() for _ in range(6)]
Xtr = np.vstack([pixel_features(i) for i, _ in train])
ytr = np.concatenate([m.ravel() for _, m in train])
clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=0).fit(Xtr, ytr)
 
test_img, test_mask = make_image()
pred = clf.predict(pixel_features(test_img)).reshape(test_mask.shape).astype(bool)
thresh = test_img > test_img.mean()                # naive baseline: global threshold
 
print("Training pixels:", len(ytr), "| features per pixel:", Xtr.shape[1])
print(f"Global threshold -> Dice = {dice(thresh, test_mask):.3f}, IoU = {iou(thresh, test_mask):.3f}")
print(f"Pixel classifier -> Dice = {dice(pred, test_mask):.3f}, IoU = {iou(pred, test_mask):.3f}")
 
fig, ax = plt.subplots(1, 4, figsize=(13, 3.5))
for a, (im, t) in zip(ax, [(test_img, "Input image"), (test_mask, "Ground truth"),
                           (thresh, "Global threshold"), (pred, "Pixel-wise network")]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig("fig4_2.png", dpi=300)
