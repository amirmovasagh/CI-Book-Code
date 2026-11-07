# Example 1.1: Binary classification with logistic regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
 
# 1) Create synthetic data: two groups of points in the plane
X, y = make_blobs(n_samples=200, centers=[(-2, 2), (2, -1)],
                  cluster_std=1.3, random_state=42)
 
# 2) Train the classifier
model = LogisticRegression()
model.fit(X, y)
 
# 3) Evaluate and predict for a new point
print("Accuracy on training data:", round(model.score(X, y), 3))
new_point = np.array([[0.5, 0.5]])
print("Predicted class for (0.5, 0.5):", model.predict(new_point)[0])
print("Class probabilities:", model.predict_proba(new_point).round(3))
 
# 4) Plot the data and the decision boundary (separating line)
w1, w2 = model.coef_[0]
b = model.intercept_[0]
xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
plt.figure(figsize=(6, 4.5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#F2B701", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#3C6E8F", label="Class 1")
plt.plot(xs, -(w1 * xs + b) / w2, "k--", label="Decision boundary")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.legend(); plt.title("Binary classification")
plt.tight_layout(); plt.savefig("fig1_1.png", dpi=300)
