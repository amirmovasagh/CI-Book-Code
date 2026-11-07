# Example 1.10: Underfitting, appropriate fit, and overfitting with polynomial regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
 
rng = np.random.default_rng(3)
x = np.sort(rng.uniform(0, 1, 15))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, 15)      # training data
x_val = np.sort(rng.uniform(0, 1, 30))
y_val = np.sin(2 * np.pi * x_val) + rng.normal(0, 0.2, 30)  # validation data
 
xs = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, degree, title in zip(axes, [1, 4, 15],
                             ["Underfitting", "Balanced", "Overfitting"]):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)
    tr = mean_squared_error(y, model.predict(x.reshape(-1, 1)))
    va = mean_squared_error(y_val, model.predict(x_val.reshape(-1, 1)))
    print(f"degree={degree:>2} | train MSE={tr:.3f} | validation MSE={va:.3f}")
    ax.scatter(x, y, c="#3C6E8F", s=18, label="train")
    ax.plot(xs, model.predict(xs.reshape(-1, 1)), "r", lw=2)
    ax.set_ylim(-2, 2); ax.set_title(f"{title} (degree={degree})")
plt.tight_layout(); plt.savefig("fig1_5.png", dpi=300)
