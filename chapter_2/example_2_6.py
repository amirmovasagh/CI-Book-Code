# Example 2.6: Gradient descent on a simple cost function and the effect of the learning rate
import numpy as np
import matplotlib.pyplot as plt
 
J  = lambda w: (w - 3) ** 2 + 1        # cost function, minimum at w = 3
dJ = lambda w: 2 * (w - 3)             # its derivative (gradient)
 
def gradient_descent(w0, lr, steps=15):
    path = [w0]
    for _ in range(steps):
        w0 = w0 - lr * dJ(w0)          # w <- w - η * dJ/dw
        path.append(w0)
    return np.array(path)
 
ws = np.linspace(-2, 8, 300)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, lr in zip(axes, [0.05, 0.3, 1.02]):
    p = gradient_descent(-1.0, lr)
    print(f"lr={lr:<5} final w = {p[-1]:9.3f} | final J = {J(p[-1]):10.3f}")
    ax.plot(ws, J(ws), c="#4a90d9")
    ax.plot(p, J(p), "o-", c="#d1495b", ms=4)
    ax.set_ylim(0, 40); ax.set_title(f"learning rate = {lr}")
    ax.set_xlabel("w"); ax.set_ylabel("J(w)")
plt.tight_layout(); plt.savefig("fig2_2.png", dpi=300)
