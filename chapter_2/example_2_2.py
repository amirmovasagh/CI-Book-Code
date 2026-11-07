# Example 2.2: Common activation functions and their plots
import numpy as np
import matplotlib.pyplot as plt
 
def step(z):        return np.where(z >= 0, 1.0, 0.0)
def linear(z):      return z
def sigmoid(z):     return 1 / (1 + np.exp(-z))
def tanh(z):        return np.tanh(z)
def relu(z):        return np.maximum(0, z)
def leaky_relu(z, a=0.1): return np.where(z > 0, z, a * z)
def elu(z, a=1.0):  return np.where(z > 0, z, a * (np.exp(z) - 1))
 
funcs = [("Binary step", step), ("Linear", linear), ("Sigmoid", sigmoid),
         ("Tanh", tanh), ("ReLU", relu), ("Leaky ReLU", leaky_relu), ("ELU", elu)]
 
z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("z =", z)
for name, f in funcs:
    print(f"{name:<12}:", np.round(f(z), 3))
 
zz = np.linspace(-4, 4, 400)
fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, (name, f) in zip(axes.ravel(), funcs):
    ax.plot(zz, f(zz), lw=2.5, color="#1b9e9e")
    ax.axhline(0, c="gray", lw=0.6); ax.axvline(0, c="gray", lw=0.6)
    ax.set_title(name); ax.grid(alpha=0.3)
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig("fig2_1.png", dpi=300)
