# Example 5.9: A decision tree for breast cancer diagnosis, displaying the tree and feature importances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
 
data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.25,
                                          stratify=data.target, random_state=0)
for depth in (1, 3, None):
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"max_depth={str(depth):<5} leaves={t.get_n_leaves():>3} "
          f"train={t.score(X_tr, y_tr):.3f} test={t.score(X_te, y_te):.3f}")
 
tree = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0).fit(X_tr, y_tr)
print(export_text(tree, feature_names=list(data.feature_names), decimals=2))
imp = sorted(zip(tree.feature_importances_, data.feature_names), reverse=True)[:4]
print("Top features:", [(str(n), round(float(v), 3)) for v, n in imp])
 
plt.figure(figsize=(14, 6))
plot_tree(tree, feature_names=data.feature_names, class_names=["malignant", "benign"],
          filled=True, rounded=True, fontsize=8)
plt.tight_layout(); plt.savefig("fig5_6.png", dpi=300)
