# Example B.1: The confusion matrix and the sensitivity, specificity, precision, and F1 metrics
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
 
# 1 = disease, 0 = healthy (a small screening example)
y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
 
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
print("Accuracy   :", accuracy_score(y_true, y_pred))
print("Sensitivity:", recall_score(y_true, y_pred), "(= TP / (TP + FN))")
print("Specificity:", round(tn / (tn + fp), 3), "(= TN / (TN + FP))")
print("Precision  :", round(precision_score(y_true, y_pred), 3), "(= TP / (TP + FP))")
print("F1 score   :", round(f1_score(y_true, y_pred), 3))
always_healthy = np.zeros_like(y_true)
print("A model that always says 'healthy' -> accuracy:", accuracy_score(y_true, always_healthy),
      "| sensitivity:", recall_score(y_true, always_healthy))
