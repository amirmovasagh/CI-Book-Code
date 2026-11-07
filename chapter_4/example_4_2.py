# Example 4.2: A complete convolutional network from scratch with NumPy
# Input 8x8 -> Conv(3x3, 8 filters, same) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(10) -> Softmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
 
rng = np.random.default_rng(0)
digits = load_digits()                                # 1797 handwritten digits, 8x8 pixels
X = digits.images / 16.0
X_tr, X_te, y_tr, y_te = train_test_split(X, digits.target, test_size=0.25,
                                          stratify=digits.target, random_state=0)
 
def im2col(x, k=3, p=1):
    """(N, H, W) -> (N, H*W, k*k): every k x k patch becomes one row."""
    N, H, W = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    cols = np.empty((N, H * W, k * k))
    for i in range(H):
        for j in range(W):
            cols[:, i * W + j, :] = xp[:, i:i + k, j:j + k].reshape(N, -1)
    return cols
 
F = 8                                                 # number of filters
Wc = rng.normal(0, 0.3, (9, F)); bc = np.zeros(F)     # conv filters (3x3 = 9 weights each)
Wd = rng.normal(0, 0.1, (4 * 4 * F, 10)); bd = np.zeros(10)
 
def forward(x):
    cols = im2col(x)                                  # (N, 64, 9)
    z = cols @ Wc + bc                                # convolution  (N, 64, F)
    a = np.maximum(z, 0).reshape(-1, 8, 8, F)         # ReLU
    blocks = a.reshape(-1, 4, 2, 4, 2, F).transpose(0, 1, 3, 5, 2, 4).reshape(-1, 4, 4, F, 4)
    pooled = blocks.max(-1)                           # max pooling 2x2 -> (N, 4, 4, F)
    flat = pooled.reshape(len(x), -1)
    logits = flat @ Wd + bd
    e = np.exp(logits - logits.max(1, keepdims=True))
    return cols, z, blocks, flat, e / e.sum(1, keepdims=True)
 
lr, history = 0.1, []
for epoch in range(40):
    order = rng.permutation(len(X_tr))
    for start in range(0, len(order), 32):            # mini-batches of 32
        idx = order[start:start + 32]
        x, t = X_tr[idx], y_tr[idx]
        cols, z, blocks, flat, prob = forward(x)
        n = len(x)
        d_logits = prob.copy(); d_logits[np.arange(n), t] -= 1; d_logits /= n
        dWd = flat.T @ d_logits; dbd = d_logits.sum(0)
        d_pooled = (d_logits @ Wd.T).reshape(n, 4, 4, F)
        mask = blocks == blocks.max(-1, keepdims=True)            # where the max came from
        d_blocks = mask * d_pooled[..., None]
        d_a = d_blocks.reshape(n, 4, 4, F, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(n, 64, F)
        d_z = d_a * (z > 0)                                       # back through ReLU
        dWc = np.einsum("npk,npf->kf", cols, d_z); dbc = d_z.sum((0, 1))
        Wd -= lr * dWd; bd -= lr * dbd; Wc -= lr * dWc; bc -= lr * dbc
    acc_tr = np.mean(forward(X_tr)[-1].argmax(1) == y_tr)
    acc_te = np.mean(forward(X_te)[-1].argmax(1) == y_te)
    history.append((acc_tr, acc_te))
    if epoch % 10 == 9:
        print(f"epoch {epoch + 1:>2}: train acc = {acc_tr:.3f} | test acc = {acc_te:.3f}")
 
pred = forward(X_te)[-1].argmax(1)
print("Trainable parameters:", Wc.size + bc.size + Wd.size + bd.size)
print("Confusion matrix (rows = true digit):\n", confusion_matrix(y_te, pred))
 
fig, ax = plt.subplots(2, 9, figsize=(13, 3.4))
ax[0, 0].imshow(X_te[0], cmap="gray"); ax[0, 0].set_title(f"input: {y_te[0]}", fontsize=9)
fmaps = np.maximum(im2col(X_te[:1]) @ Wc + bc, 0).reshape(8, 8, F)
for f in range(F):
    ax[0, f + 1].imshow(Wc[:, f].reshape(3, 3), cmap="bwr"); ax[0, f + 1].set_title(f"filter {f}", fontsize=9)
    ax[1, f + 1].imshow(fmaps[:, :, f], cmap="viridis"); ax[1, f + 1].set_title(f"feature map {f}", fontsize=8)
ax[1, 0].plot([h[0] for h in history], label="train"); ax[1, 0].plot([h[1] for h in history], label="test")
ax[1, 0].set_title("accuracy", fontsize=9); ax[1, 0].legend(fontsize=6)
for a in ax.ravel()[1:]:
    if a is not ax[1, 0]: a.axis("off")
ax[0, 0].axis("off")
plt.tight_layout(); plt.savefig("fig4_1.png", dpi=300)
