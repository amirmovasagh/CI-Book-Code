# Example 5.8: Computing entropy, Gini impurity, and information gain by hand
import numpy as np
 
def entropy(counts):
    p = np.array(counts) / sum(counts); p = p[p > 0]
    return -(p * np.log2(p)).sum()
 
def gini(counts):
    p = np.array(counts) / sum(counts)
    return 1 - (p ** 2).sum()
 
parent = [9, 5]                       # 9 sick, 5 healthy patients
left, right = [6, 1], [3, 4]          # split by "fever = yes / no"
n = sum(parent)
ig = entropy(parent) - (sum(left) / n * entropy(left) + sum(right) / n * entropy(right))
print("Entropy parent:", round(entropy(parent), 3), "| Gini parent:", round(gini(parent), 3))
print("Entropy left/right:", round(entropy(left), 3), round(entropy(right), 3))
print("Information gain of the split:", round(ig, 3))
