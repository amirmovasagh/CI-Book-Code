# Example 4.4: K-fold cross-validation — reporting accuracy in the style of medical papers
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
 
X, y = load_digits(return_X_y=True)
model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0)
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
scores = cross_val_score(model, X / 16, y, cv=cv)
for i, s in enumerate(scores, 1):
    print(f"Fold {i}: accuracy = {100 * s:.1f}%")
print(f"Average: {100 * scores.mean():.1f}%  (± {100 * scores.std():.1f})")
