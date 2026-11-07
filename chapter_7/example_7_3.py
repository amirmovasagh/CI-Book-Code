# Example 7.3: Particle swarm optimization (PSO) — inspired by the collective movement of bird flocks
import numpy as np
 
def rastrigin(p):  # a classic test function with many local minima, global minimum 0 at (0,0)
    return 10 * p.shape[-1] + (p ** 2 - 10 * np.cos(2 * np.pi * p)).sum(-1)
 
rng = np.random.default_rng(1)
N, D, iters, w, c1, c2 = 30, 2, 100, 0.7, 1.5, 1.5
pos = rng.uniform(-5, 5, (N, D)); vel = np.zeros((N, D))
pbest, pbest_val = pos.copy(), rastrigin(pos)
gbest = pbest[pbest_val.argmin()].copy()
for t in range(iters):
    r1, r2 = rng.random((N, D)), rng.random((N, D))
    vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)   # personal + social pull
    pos = np.clip(pos + vel, -5, 5)
    val = rastrigin(pos)
    better = val < pbest_val
    pbest[better], pbest_val[better] = pos[better], val[better]
    gbest = pbest[pbest_val.argmin()].copy()
    if t in (0, 9, 49, 99): print(f"iteration {t + 1:>3}: best value = {pbest_val.min():.5f} at {gbest.round(4)}")
