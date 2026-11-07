# Example 2.4: Single-layer perceptron — succeeds on AND, fails on XOR
import numpy as np
 
def train_perceptron(X, y, epochs=20, lr=0.1):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        for xi, ti in zip(X, y):
            pred = 1 if xi @ w + b >= 0 else 0
            w += lr * (ti - pred) * xi          # perceptron learning rule
            b += lr * (ti - pred)
    return w, b
 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for name, y in [("AND", np.array([0, 0, 0, 1])), ("XOR", np.array([0, 1, 1, 0]))]:
    w, b = train_perceptron(X, y)
    pred = (X @ w + b >= 0).astype(int)
    print(f"{name}: targets {y} predictions {pred} accuracy {np.mean(pred == y):.2f}")
