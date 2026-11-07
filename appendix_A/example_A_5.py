# Example A.5: Plotting with Matplotlib
import numpy as np
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(1)
t = np.linspace(0, 2, 800)
ecg_like = np.sin(2 * np.pi * 1.2 * t) ** 63 + 0.05 * rng.normal(size=t.size)   # toy "ECG" spikes
values = rng.normal(120, 12, 500)
 
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].plot(t, ecg_like, c="#d1495b"); ax[0].set_title("Line plot (toy signal)"); ax[0].set_xlabel("time (s)")
ax[1].hist(values, bins=30, color="#3C6E8F", edgecolor="white"); ax[1].set_title("Histogram (systolic BP)")
ax[2].scatter(rng.uniform(20, 80, 80), rng.uniform(60, 120, 80), c="#2a9d8f"); ax[2].set_title("Scatter plot")
ax[2].set_xlabel("age"); ax[2].set_ylabel("weight")
plt.tight_layout(); plt.savefig("figA_1.png", dpi=300)
print("Figure saved with", len(fig.axes), "subplots")
