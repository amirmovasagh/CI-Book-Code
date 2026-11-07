# Example B.4: Imbalanced data — the effect of class weighting
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, accuracy_score
 
X, y = make_classification(n_samples=5000, n_features=10, n_informative=4, weights=[0.97, 0.03],
                           class_sep=0.8, random_state=3)
print("Positive (disease) share:", round(y.mean(), 3))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
for cw in (None, "balanced"):
    m = LogisticRegression(class_weight=cw, max_iter=1000).fit(X_tr, y_tr)
    pr = m.predict(X_te)
    print(f"class_weight={str(cw):<9} accuracy={accuracy_score(y_te, pr):.3f}  "
          f"sensitivity={recall_score(y_te, pr):.3f}  precision={precision_score(y_te, pr):.3f}  "
          f"balanced acc={balanced_accuracy_score(y_te, pr):.3f}")
