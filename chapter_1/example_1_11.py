# Example 1.11: Error curve versus epoch and early stopping
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
 
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, (40, 1))
y = np.sin(3 * X).ravel() + rng.normal(0, 0.25, 40)
X_val = rng.uniform(-1, 1, (200, 1))
y_val = np.sin(3 * X_val).ravel() + rng.normal(0, 0.25, 200)
 
# A network deliberately larger than needed, so that overfitting occurs
net = MLPRegressor(hidden_layer_sizes=(200, 200), learning_rate_init=0.002,
                   alpha=0.0, random_state=1)
train_loss, val_loss = [], []
for epoch in range(1500):
    net.partial_fit(X, y)                     # one training epoch
    train_loss.append(mean_squared_error(y, net.predict(X)))
    val_loss.append(mean_squared_error(y_val, net.predict(X_val)))
 
best_epoch = int(np.argmin(val_loss))
print("Best epoch (minimum validation error):", best_epoch)
print("Validation MSE at best epoch:", round(val_loss[best_epoch], 4))
print("Validation MSE at last epoch:", round(val_loss[-1], 4))
print("Train MSE at last epoch:", round(train_loss[-1], 4))
 
plt.figure(figsize=(7, 4.2))
plt.plot(train_loss, label="training")
plt.plot(val_loss, label="validation")
plt.axvline(best_epoch, ls="--", c="k")
plt.text(best_epoch, max(val_loss) * 0.8, "  early stopping")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss (MSE)")
plt.legend(); plt.tight_layout(); plt.savefig("fig1_6.png", dpi=300)
 
# In practice, sklearn does this automatically:
auto = MLPRegressor(hidden_layer_sizes=(200, 200), early_stopping=True,
                    validation_fraction=0.2, n_iter_no_change=20,
                    max_iter=2000, random_state=1).fit(X, y)
print("sklearn early_stopping stopped after", auto.n_iter_, "epochs")
