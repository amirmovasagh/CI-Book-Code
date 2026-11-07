# Example 6.3: Continuous membership functions — triangular, trapezoidal, Gaussian, bell, and sigmoidal
import numpy as np
import matplotlib.pyplot as plt
 
def trimf(x, a, b, c):     return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)
def trapmf(x, a, b, c, d): return np.maximum(np.minimum.reduce([(x - a) / (b - a), np.ones_like(x), (d - x) / (d - c)]), 0)
def gaussmf(x, c, s):      return np.exp(-((x - c) ** 2) / (2 * s ** 2))
def gbellmf(x, a, b, c):   return 1 / (1 + np.abs((x - c) / a) ** (2 * b))
def sigmf(x, a, c):        return 1 / (1 + np.exp(-a * (x - c)))
 
x = np.linspace(-1, 3, 401)
mfs = {"Triangular (0.5, 1, 1.5)": trimf(x, 0.5, 1, 1.5),
       "Trapezoidal (0, 0.7, 1.3, 2)": trapmf(x, 0, 0.7, 1.3, 2),
       "Gaussian (c=1, σ=0.4)": gaussmf(x, 1, 0.4),
       "Generalized bell (0.5, 2, 1)": gbellmf(x, 0.5, 2, 1),
       "Sigmoid (a=6, c=1)": sigmf(x, 6, 1)}
for name, m in mfs.items():
    print(f"{name:<30} μ(1.2) = {np.interp(1.2, x, m):.3f}")
 
plt.figure(figsize=(8, 3.8))
for name, m in mfs.items(): plt.plot(x, m, lw=2, label=name)
plt.xlabel("x"); plt.ylabel("μ(x)"); plt.title('"numbers close to 1" — different membership shapes')
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("fig6_1.png", dpi=300)
