# Example 7.4: ANFIS from scratch — a Sugeno fuzzy system that learns its parameters from data
# Hybrid learning: consequents by least squares, premise (membership) params by gradient descent
import numpy as np
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(-3, 3, 120)); y = np.sin(1.5 * x) + 0.3 * x + rng.normal(0, 0.05, x.size)
 
M = 4                                                     # number of fuzzy rules / membership functions
c = np.linspace(-3, 3, M); s = np.full(M, 0.45)        # Layer 1: initial (too narrow) Gaussian MFs
 
def layers(x, c, s):
    mu = np.exp(-((x[:, None] - c) ** 2) / (2 * s ** 2))  # Layer 1: fuzzification
    w = mu                                                # Layer 2: rule firing strength
    wn = w / w.sum(1, keepdims=True)                      # Layer 3: normalization
    A = np.hstack([wn * x[:, None], wn])                  # Layer 4 design matrix: wn*(p*x + r)
    return mu, wn, A
 
lr, hist = 0.3, []
for epoch in range(600):
    mu, wn, A = layers(x, c, s)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)          # forward pass: least squares for p, r
    p, r = theta[:M], theta[M:]
    f = p * x[:, None] + r                                # rule outputs
    out = (wn * f).sum(1)                                 # Layer 5: weighted sum
    err = out - y; hist.append(np.sqrt(np.mean(err ** 2)))
    # backward pass: gradient of MSE w.r.t. c and s
    d_wn = err[:, None] * (f - out[:, None]) / mu.sum(1, keepdims=True)
    d_c = (d_wn * mu * (x[:, None] - c) / s ** 2).mean(0)
    d_s = (d_wn * mu * (x[:, None] - c) ** 2 / s ** 3).mean(0)
    c -= lr * d_c; s = np.maximum(s - lr * d_s, 0.2)
 
print(f"RMSE after epoch 1: {hist[0]:.4f} | after epoch 600: {hist[-1]:.4f}")
print("Learned MF centers:", c.round(2)); print("Learned MF widths :", s.round(2))
print("Rule consequents (p, r):", [(round(float(a), 2), round(float(b), 2)) for a, b in zip(p, r)])
 
xs = np.linspace(-3, 3, 300); mu_s, wn_s, A_s = layers(xs, c, s)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].scatter(x, y, s=8, c="gray", label="data"); ax[0].plot(xs, A_s @ theta, "r", lw=2, label="ANFIS output")
ax[0].legend(); ax[0].set_title("ANFIS fit")
ax[1].plot(xs, mu_s); ax[1].set_title("Learned membership functions")
plt.tight_layout(); plt.savefig("fig7_2.png", dpi=300)
