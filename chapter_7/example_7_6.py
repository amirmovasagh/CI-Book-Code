# Example 7.6: Tuning an SVM for breast cancer diagnosis — grid search, random search, and differential evolution
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

X, y = load_breast_cancer(return_X_y=True)                  # 0 = malignant, 1 = benign
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
lo, hi = np.array([-2.0, -5.0]), np.array([3.0, 0.0])        # search space: log10(C), log10(gamma)

def make_model(p):
    return make_pipeline(StandardScaler(), SVC(C=10 ** p[0], gamma=10 ** p[1]))

def cv_score(p):                                             # the fitness: mean 5-fold CV accuracy
    return cross_val_score(make_model(p), X_tr, y_tr, cv=cv).mean()

def random_search(n, seed):
    pts = np.random.default_rng(seed).uniform(lo, hi, (n, 2))
    sc = np.array([cv_score(p) for p in pts])
    return pts[sc.argmax()], np.maximum.accumulate(sc)

def de_search(N, gens, seed, F=0.6, CR=0.7):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(lo, hi, (N, 2)); fit = np.array([cv_score(p) for p in pop])
    trace, snapshots = list(np.maximum.accumulate(fit)), [pop.copy()]
    for g in range(gens):
        for i in range(N):
            a, b, c = pop[rng.choice([j for j in range(N) if j != i], 3, replace=False)]
            trial = np.where(rng.random(2) < CR, np.clip(a + F * (b - c), lo, hi), pop[i])
            ft = cv_score(trial)
            if ft >= fit[i]: pop[i], fit[i] = trial, ft          # greedy selection (we maximize)
            trace.append(fit.max())
        snapshots.append(pop.copy())
    return pop[fit.argmax()], np.array(trace), snapshots

# 1) grid search: 11 x 11 = 121 evaluations
gC, gG = np.linspace(lo[0], hi[0], 11), np.linspace(lo[1], hi[1], 11)
grid = np.array([[cv_score([c, g]) for c in gC] for g in gG])
i, j = np.unravel_index(grid.argmax(), grid.shape); best_grid = np.array([gC[j], gG[i]])
print(f"default SVC (C=1, gamma='scale'): CV acc = {cross_val_score(make_pipeline(StandardScaler(), SVC()), X_tr, y_tr, cv=cv).mean():.4f}")
print(f"grid search   (121 evals): C = {10 ** best_grid[0]:.3f}, gamma = {10 ** best_grid[1]:.5f}, CV acc = {grid.max():.4f}")

# 2) random search and 3) DE, each with a budget of 40 evaluations, repeated with 5 seeds
rs = [random_search(40, s) for s in range(5)]
de = [de_search(8, 4, s) for s in range(5)]
print("random search (40 evals): best CV acc per seed =", np.round([r[1][-1] for r in rs], 4))
print("DE            (40 evals): best CV acc per seed =", np.round([d[1][-1] for d in de], 4))
print(f"one misclassified sample changes the mean CV accuracy by 1/{len(y_tr)} = {1 / len(y_tr):.4f}")

# the test set is used once, for the configuration chosen on the training data
model = make_model(best_grid).fit(X_tr, y_tr)
tn, fp, fn, tp = confusion_matrix(y_te, model.predict(X_te), labels=[1, 0]).ravel()   # positive class = malignant
print(f"test set ({len(y_te)} patients): accuracy = {(tp + tn) / len(y_te):.4f}, "
      f"sensitivity = {tp / (tp + fn):.3f}, specificity = {tn / (tn + fp):.3f}, missed cancers = {fn}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
im = ax[0].imshow(grid, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], aspect="auto", cmap="viridis", vmin=0.6)
plt.colorbar(im, ax=ax[0], label="5-fold CV accuracy")
for k, P in enumerate(de[0][2]):
    ax[0].scatter(P[:, 0], P[:, 1], s=20, color=plt.cm.autumn(k / 4), edgecolor="k", lw=0.4,
                  label="DE population, generations 0 to 4" if k == 0 else None)
ax[0].scatter(*best_grid, marker="*", s=220, c="white", edgecolor="k", label="best grid point")
ax[0].set_xlabel("log10(C)"); ax[0].set_ylabel("log10(gamma)"); ax[0].legend(loc="lower left", fontsize=8)
ax[0].set_title("Accuracy landscape (grid) and one DE run")
n = np.arange(1, 41)
ax[1].plot(n, np.mean([r[1] for r in rs], 0), "o-", ms=3, label="random search (mean of 5 seeds)")
ax[1].plot(n, np.mean([d[1] for d in de], 0), "s-", ms=3, label="DE (mean of 5 seeds)")
ax[1].axhline(grid.max(), ls="--", color="gray", label="grid search, 121 evaluations")
ax[1].set_xlabel("number of evaluations"); ax[1].set_ylabel("best CV accuracy so far"); ax[1].set_ylim(0.95, 0.985)
ax[1].legend(fontsize=8); ax[1].set_title("Progress of the search")
plt.tight_layout(); plt.savefig("fig7_6.png", dpi=300)
