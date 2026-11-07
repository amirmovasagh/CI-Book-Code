# Example 5.11: Beyond the three algorithms — SVM and random forest on the breast cancer data
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
 
data = load_breast_cancer(); X, y = data.data, data.target
tree = DecisionTreeClassifier(random_state=0)
forest = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
for name, m in [("single decision tree", tree), ("random forest (300 trees)", forest), ("SVM (RBF kernel)", svm)]:
    s = cross_val_score(m, X, y, cv=5)
    print(f"{name:<26} 5-fold accuracy = {s.mean():.3f} ± {s.std():.3f}")
 
grid = GridSearchCV(svm, {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}, cv=5).fit(X, y)
print("Best SVM hyperparameters:", grid.best_params_, "| CV accuracy:", round(grid.best_score_, 3))
 
forest.fit(X, y)
top = np.argsort(forest.feature_importances_)[::-1][:5]
print("Random forest top-5 features:")
for i in top:
    print(f"   {data.feature_names[i]:<22} {forest.feature_importances_[i]:.3f}")
