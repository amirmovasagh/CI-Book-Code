# Example 6.4: The fuzzy set "tall" from three perspectives, and the linguistic variable "height"
import numpy as np
import matplotlib.pyplot as plt
 
sig = lambda x, a, c: 1 / (1 + np.exp(-a * (x - c)))
h = np.linspace(140, 200, 601)
 
# Part 1: "tall" is context dependent (three different observers / populations)
tall_views = {"view 1 (children)": sig(h, 0.4, 163), "view 2 (general)": sig(h, 0.35, 171),
              "view 3 (basketball)": sig(h, 0.9, 186)}
i175 = np.argmin(abs(h - 175))
for name, m in tall_views.items():
    print(f"μ_tall(175) from {name:<20} = {m[i175]:.2f}")
 
# Part 2: linguistic variable "height" with three terms
short = 1 - sig(h, 0.45, 158)
medium = sig(h, 0.45, 158) * (1 - sig(h, 0.45, 178))
tall = sig(h, 0.45, 178)
for x in (150, 165, 175, 185):
    i = np.argmin(abs(h - x))
    print(f"height {x} cm -> short={short[i]:.2f}, medium={medium[i]:.2f}, tall={tall[i]:.2f}")
 
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for (n, m), c in zip(tall_views.items(), ["#4a90d9", "#d1495b", "#2a9d8f"]):
    ax[0].plot(h, m, lw=2.5, c=c, label=n)
ax[0].axvline(175, ls="--", c="gray"); ax[0].set_title('Fuzzy set "tall" from three viewpoints')
ax[0].set_xlabel("height (cm)"); ax[0].legend(fontsize=8)
for m, n, c in [(short, "short", "#4a90d9"), (medium, "medium", "#d1495b"), (tall, "tall", "#2a9d8f")]:
    ax[1].plot(h, m, lw=2.5, c=c, label=n)
ax[1].set_title('Linguistic variable "height"'); ax[1].set_xlabel("height (cm)"); ax[1].legend()
plt.tight_layout(); plt.savefig("fig6_2.png", dpi=300)
