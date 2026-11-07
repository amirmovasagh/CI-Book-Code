# Example 2.3: The Softmax function — converting layer outputs into probabilities
import numpy as np
 
def softmax(z):
    e = np.exp(z - np.max(z))     # subtract max for numerical stability
    return e / e.sum()
 
logits = np.array([1.3, 5.1, 2.2, 0.7, 1.1])   # raw output of the last layer
p = softmax(logits)
print("Logits       :", logits)
print("Probabilities:", p.round(2))
print("Sum of probs :", round(p.sum(), 6))
print("Predicted class index:", int(np.argmax(p)))
