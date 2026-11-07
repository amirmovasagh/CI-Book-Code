# Example 7.5: Differential evolution (DE/rand/1/bin) from scratch, compared with PSO on the 10-D Rastrigin function
import numpy as np
import matplotlib.pyplot as plt

def rastrigin(p):  # global minimum 0 at the origin
    return 10 * p.shape[-1] + (p ** 2 - 10 * np.cos(2 * np.pi * p)).sum(-1)

def de(f, D, N=40, F=0.5, CR=0.9, gens=500, lo=-5.12, hi=5.12, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, (N, D)); fit = f(pop); history = [fit.min()]
    for g in range(gens):
        for i in range(N):
            a, b, c = pop[rng.choice([j for j in range(N) if j != i], 3, replace=False)]
            mutant = np.clip(a + F * (b - c), lo, hi)            # mutation: a difference vector
            cross = rng.random(D) < CR
            cross[rng.integers(D)] = True                        # at least one gene from the mutant
            trial = np.where(cross, mutant, pop[i])               # binomial crossover
            ft = f(trial)
            if ft <= fit[i]:                                      # greedy one-to-one selection
                pop[i], fit[i] = trial, ft
        history.append(fit.min())
    return pop[fit.argmin()], fit.min(), history

def pso(f, D, N=40, w=0.7, c1=1.5, c2=1.5, iters=500, lo=-5.12, hi=5.12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, (N, D)); v = np.zeros((N, D))
    pb, pbv = x.copy(), f(x); gb = pb[pbv.argmin()].copy(); history = [pbv.min()]
    for t in range(iters):
        r1, r2 = rng.random((N, D)), rng.random((N, D))
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
        x = np.clip(x + v, lo, hi); val = f(x)
        better = val < pbv; pb[better], pbv[better] = x[better], val[better]
        gb = pb[pbv.argmin()].copy(); history.append(pbv.min())
    return gb, pbv.min(), history

D = 10
algs = {"DE, CR=0.1": lambda s: de(rastrigin, D, CR=0.1, seed=s),
        "DE, CR=0.9": lambda s: de(rastrigin, D, CR=0.9, seed=s),
        "PSO":        lambda s: pso(rastrigin, D, seed=s)}
results = {name: [alg(s) for s in range(5)] for name, alg in algs.items()}
for name, runs in results.items():
    best = np.array([r[1] for r in runs])
    print(f"{name:<11}: best value over 5 runs  mean = {best.mean():7.3f}   min = {best.min():7.3f}   max = {best.max():7.3f}")
print("evaluations per run:", 40 * 501)

plt.figure(figsize=(6.5, 3.2))
for name, col in [("DE, CR=0.1", "#2A9D8F"), ("DE, CR=0.9", "#1F4E79"), ("PSO", "#E76F51")]:
    for k, r in enumerate(results[name]):
        plt.semilogy(np.maximum(r[2], 1e-8), color=col, alpha=0.8, lw=1.4, label=name if k == 0 else None)
plt.xlabel("generation / iteration"); plt.ylabel("best value (log scale)")
plt.title("10-D Rastrigin: 5 runs of each algorithm (values below 1e-8 drawn at 1e-8)"); plt.legend()
plt.tight_layout(); plt.savefig("fig7_5.png", dpi=300)
