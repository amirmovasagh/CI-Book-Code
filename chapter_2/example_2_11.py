# Example 2.11: L2 regularization — taming the overfitting of a degree-15 polynomial model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
 
rng = np.random.default_rng(3)
x = np.sort(rng.uniform(0, 1, 15)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 15)
xv = rng.uniform(x.min(), x.max(), 200); yv = np.sin(2 * np.pi * xv) + rng.normal(0, 0.2, 200)
xs = np.linspace(x.min(), x.max(), 300)
 
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, lam in zip(ax, [1e-9, 1e-2, 10.0]):
    model = make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=lam))
    model.fit(x[:, None], y)
    tr = mean_squared_error(y, model.predict(x[:, None])); va = mean_squared_error(yv, model.predict(xv[:, None]))
    w = model[-1].coef_
    print(f"lambda={lam:<6g} train MSE={tr:.3f} | validation MSE={va:9.3f} | ||w||^2 = {np.sum(w**2):12.1f}")
    a.scatter(x, y, s=18, c="#3C6E8F"); a.plot(xs, model.predict(xs[:, None]), "r", lw=2)
    a.plot(xs, np.sin(2 * np.pi * xs), "k--", lw=1); a.set_ylim(-2, 2); a.set_title(f"lambda = {lam:g}")
plt.tight_layout(); plt.savefig("fig2_6.png", dpi=300)
