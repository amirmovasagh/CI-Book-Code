# Example 2.8: Comparing optimizers — SGD, Momentum, and Adam on a cost surface
import numpy as np
import matplotlib.pyplot as plt
 
f  = lambda x, y: 0.1 * x**2 + 2 * y**2           # elongated bowl
gf = lambda x, y: np.array([0.2 * x, 4 * y])
 
def run(opt, steps=60, lr=0.4):
    p = np.array([-9.0, 2.0]); v = np.zeros(2); m = np.zeros(2); s = np.zeros(2)
    path = [p.copy()]
    for t in range(1, steps + 1):
        g = gf(*p)
        if opt == "SGD":
            p -= lr * 0.5 * g
        elif opt == "Momentum":
            v = 0.9 * v + g;  p -= lr * 0.1 * v
        elif opt == "Adam":
            m = 0.9 * m + 0.1 * g;  s = 0.999 * s + 0.001 * g**2
            mh, sh = m / (1 - 0.9**t), s / (1 - 0.999**t)
            p -= lr * mh / (np.sqrt(sh) + 1e-8)
        path.append(p.copy())
    return np.array(path)
 
xs, ys = np.meshgrid(np.linspace(-10, 3, 200), np.linspace(-3, 3, 200))
plt.figure(figsize=(8, 4))
plt.contour(xs, ys, f(xs, ys), levels=20, cmap="Blues")
for opt, c in [("SGD", "#d1495b"), ("Momentum", "#2a9d8f"), ("Adam", "#e9a03b")]:
    path = run(opt)
    print(f"{opt:<9} final point = {path[-1].round(3)} | f = {f(*path[-1]):.4f}")
    plt.plot(path[:, 0], path[:, 1], "o-", ms=2.5, c=c, label=opt)
plt.plot(0, 0, "k*", ms=12); plt.legend(); plt.title("Optimizer paths toward the minimum")
plt.tight_layout(); plt.savefig("fig2_5.png", dpi=300)
