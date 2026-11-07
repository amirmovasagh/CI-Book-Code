# Example 2.7: Local and global minima of a non-convex function
import numpy as np
import matplotlib.pyplot as plt
 
C  = lambda w: 0.1 * w**4 - 0.5 * w**3 - 0.2 * w**2 + 2 * w + 3
dC = lambda w: 0.4 * w**3 - 1.5 * w**2 - 0.4 * w + 2
 
def descend(w, lr=0.05, steps=200):
    path = [w]
    for _ in range(steps):
        w -= lr * dC(w); path.append(w)
    return np.array(path)
 
ws = np.linspace(-2.5, 4.8, 400)
plt.figure(figsize=(7, 4))
plt.plot(ws, C(ws), c="#4a90d9", lw=2)
for start, color in [(-2.2, "#d1495b"), (1.8, "#2a9d8f")]:
    p = descend(start)
    print(f"start w={start:>4} -> converged to w={p[-1]:.3f}, C(w)={C(p[-1]):.3f}")
    plt.plot(p, C(p), "o-", c=color, ms=3, label=f"start = {start}")
plt.xlabel("w"); plt.ylabel("C(w)"); plt.legend(); plt.title("Local vs global minimum")
plt.tight_layout(); plt.savefig("fig2_3.png", dpi=300)
