# Example 2.9: A two-layer network from scratch with backpropagation to solve the XOR problem
import numpy as np
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
 
sig = lambda z: 1 / (1 + np.exp(-z))
W1, b1 = rng.normal(0, 1, (2, 4)), np.zeros((1, 4))   # input -> hidden (4 neurons)
W2, b2 = rng.normal(0, 1, (4, 1)), np.zeros((1, 1))   # hidden -> output
lr, losses = 1.0, []
 
for epoch in range(5000):
    # --- forward propagation ---
    h = sig(X @ W1 + b1)
    out = sig(h @ W2 + b2)
    loss = np.mean((y - out) ** 2)
    losses.append(loss)
    # --- backward propagation (chain rule) ---
    d_out = (out - y) * out * (1 - out)
    d_h = (d_out @ W2.T) * h * (1 - h)
    W2 -= lr * h.T @ d_out;  b2 -= lr * d_out.sum(0, keepdims=True)
    W1 -= lr * X.T @ d_h;    b1 -= lr * d_h.sum(0, keepdims=True)
 
print("Final loss:", round(losses[-1], 5))
for xi, oi in zip(X, out):
    print(f"input {xi.astype(int)} -> output {oi[0]:.3f} -> class {int(oi[0] > 0.5)}")
 
plt.figure(figsize=(6.5, 3.8))
plt.plot(losses, c="#d1495b"); plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("MSE loss"); plt.title("Training a 2-layer network on XOR")
plt.tight_layout(); plt.savefig("fig2_4.png", dpi=300)
