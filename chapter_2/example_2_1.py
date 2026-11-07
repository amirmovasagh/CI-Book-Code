# Example 2.1: Implementing an artificial neuron from scratch
import numpy as np
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
# Three inputs (e.g., three normalized blood test markers)
x = np.array([0.8, 0.2, 0.5])
# The weight of each input and the neuron's bias
w = np.array([0.9, -0.4, 0.6])
b = -0.3
 
z = np.dot(w, x) + b          # step 1: weighted sum  (Σ wi*xi + b)
y = sigmoid(z)                # step 2: activation function
 
print("Weighted inputs w*x :", w * x)
print("Net input z         :", round(z, 3))
print("Output y = sigmoid(z):", round(y, 3))
