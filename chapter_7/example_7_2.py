# Example 7.2: Feature selection with a genetic algorithm for breast cancer diagnosis
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
data = load_breast_cancer(); X, y = data.data, data.target
rng = np.random.default_rng(0)
model = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
cache = {}
 
def fitness(mask):
    key = mask.tobytes()
    if key not in cache:
        if mask.sum() == 0: cache[key] = 0.0
        else:
            acc = cross_val_score(model, X[:, mask.astype(bool)], y, cv=5).mean()
            cache[key] = acc - 0.002 * mask.sum()       # reward accuracy, penalize many features
    return cache[key]
 
POP, GENS, n = 20, 15, X.shape[1]
pop = (rng.random((POP, n)) < 0.5).astype(int)
for g in range(GENS):
    fit = np.array([fitness(m) for m in pop])
    order = fit.argsort()[::-1]; elite = pop[order[:4]]
    kids = []
    while len(kids) < POP - 4:
        a, b = pop[rng.choice(order[:10], 2, replace=False)]
        cut = rng.integers(1, n); child = np.r_[a[:cut], b[cut:]]
        child[rng.random(n) < 0.05] ^= 1
        kids.append(child)
    pop = np.vstack([elite, kids])
    if g % 5 == 4: print(f"generation {g + 1:>2}: best fitness = {fit.max():.4f}")
 
best = pop[np.argmax([fitness(m) for m in pop])].astype(bool)
all_acc = cross_val_score(model, X, y, cv=5).mean()
sel_acc = cross_val_score(model, X[:, best], y, cv=5).mean()
print(f"All 30 features      -> CV accuracy = {all_acc:.3f}")
print(f"GA-selected {best.sum():>2} feat. -> CV accuracy = {sel_acc:.3f}")
print("Selected:", ", ".join(np.array(data.feature_names)[best][:8]), "...")
