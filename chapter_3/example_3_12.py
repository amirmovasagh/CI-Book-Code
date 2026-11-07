# Example 3.12: Hounsfield units and windowing in CT images
import numpy as np
import matplotlib.pyplot as plt
 
def mu_to_hu(mu, mu_water=0.19, mu_air=0.0):
    return 1000 * (mu - mu_water) / (mu_water - mu_air)
 
def window(hu, center, width):
    lo, hi = center - width / 2, center + width / 2
    return np.clip((hu - lo) / (hi - lo), 0, 1)
 
for tissue, mu in [("air", 0.0), ("lung", 0.05), ("fat", 0.17), ("water", 0.19),
                   ("soft tissue", 0.20), ("bone", 0.38)]:
    print(f"{tissue:<12} mu={mu:<5} -> HU = {mu_to_hu(mu):7.1f}")
 
hu = np.linspace(-1100, 1500, 600)
plt.figure(figsize=(7, 3.6))
for name, c, w in [("Lung window", -600, 1500), ("Soft-tissue window", 40, 400), ("Bone window", 400, 1800)]:
    plt.plot(hu, window(hu, c, w), lw=2, label=f"{name} (C={c}, W={w})")
plt.xlabel("Hounsfield units"); plt.ylabel("display gray level"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig3_8.png", dpi=300)
