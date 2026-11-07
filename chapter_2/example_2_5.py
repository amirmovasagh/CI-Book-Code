# Example 2.5: Computing the MSE and cross-entropy loss functions
import numpy as np
 
# Regression: actual and predicted values
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5,  0.0, 2.0, 8.0])
mse = np.mean((y_true - y_pred) ** 2)
mae = np.mean(np.abs(y_true - y_pred))
print("MSE =", mse, "| MAE =", mae)
 
# Binary classification: predicted probability of class 1
t = np.array([1, 0, 1, 1])
p_good = np.array([0.9, 0.1, 0.8, 0.95])
p_bad  = np.array([0.4, 0.7, 0.3, 0.5])
def bce(t, p): return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))
print("Binary cross-entropy (good model):", round(bce(t, p_good), 4))
print("Binary cross-entropy (bad model) :", round(bce(t, p_bad), 4))
