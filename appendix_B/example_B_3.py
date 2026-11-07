# Example B.3: Regression evaluation metrics — MAE, MSE, RMSE, and R²
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
y_true = np.array([120, 135, 128, 150, 142, 118, 160])     # measured systolic BP
y_pred = np.array([118, 130, 131, 155, 138, 121, 150])     # model predictions
mse = mean_squared_error(y_true, y_pred)
print("MAE  =", round(mean_absolute_error(y_true, y_pred), 2), "mmHg")
print("MSE  =", round(mse, 2), "mmHg²")
print("RMSE =", round(np.sqrt(mse), 2), "mmHg")
print("R²   =", round(r2_score(y_true, y_pred), 3))
print("R² of always predicting the mean =", r2_score(y_true, np.full_like(y_true, y_true.mean(), dtype=float)))
