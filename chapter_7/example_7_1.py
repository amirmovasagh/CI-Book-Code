# Example 7.1: A genetic algorithm from scratch — finding the maximum of a highly oscillating function
import numpy as np
import matplotlib.pyplot as plt
 
f = lambda x: x * np.sin(10 * np.pi * x) + 1.0          # many local maxima on [-1, 2]
rng = np.random.default_rng(42)
BITS, POP, GENS, PC, PM = 22, 50, 60, 0.8, 0.01
 
def decode(chrom):                                       # binary string -> real number in [-1, 2]
    value = chrom @ (2 ** np.arange(BITS)[::-1])
    return -1 + 3 * value / (2 ** BITS - 1)
 
pop = rng.integers(0, 2, (POP, BITS))                    # 1) random initial population
best_hist, mean_hist = [], []
for g in range(GENS):
    x = np.array([decode(c) for c in pop]); fit = f(x)   # 2) fitness evaluation
    best_hist.append(fit.max()); mean_hist.append(fit.mean())
    # 3) selection: tournament of size 3
    idx = [max(rng.choice(POP, 3, replace=False), key=lambda i: fit[i]) for _ in range(POP)]
    parents = pop[idx]
    # 4) crossover: single point
    children = parents.copy()
    for i in range(0, POP, 2):
        if rng.random() < PC:
            cut = rng.integers(1, BITS)
            children[i, cut:], children[i + 1, cut:] = parents[i + 1, cut:], parents[i, cut:]
    # 5) mutation: flip bits with small probability
    flip = rng.random(children.shape) < PM
    children[flip] = 1 - children[flip]
    children[0] = pop[fit.argmax()]                      # elitism: keep the best one
    pop = children
 
x = np.array([decode(c) for c in pop]); best = x[f(x).argmax()]
print(f"Best x found = {best:.4f}, f(x) = {f(best):.4f}")
grid = np.linspace(-1, 2, 200001); print(f"True maximum (grid search): x = {grid[f(grid).argmax()]:.4f}, f = {f(grid).max():.4f}")
print("Best fitness in generations 1, 10, 30, 60:", [round(float(best_hist[i]), 4) for i in (0, 9, 29, 59)])
 
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
xs = np.linspace(-1, 2, 2000); ax[0].plot(xs, f(xs), c="#4a90d9"); ax[0].plot(best, f(best), "r*", ms=15)
ax[0].set_title("f(x) = x·sin(10πx) + 1")
ax[1].plot(best_hist, label="best"); ax[1].plot(mean_hist, label="population mean")
ax[1].set_xlabel("generation"); ax[1].set_ylabel("fitness"); ax[1].legend()
plt.tight_layout(); plt.savefig("fig7_1.png", dpi=300)
