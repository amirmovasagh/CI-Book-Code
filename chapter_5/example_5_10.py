# Example 5.10: Final comparison of KNN, the decision tree, and K-Means on one dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
 
X, y = load_breast_cancer(return_X_y=True)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(7))
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
for name, m in [("KNN (k=7)", knn), ("Decision tree", tree)]:
    s = cross_val_score(m, X, y, cv=5)
    print(f"{name:<14} 5-fold accuracy = {s.mean():.3f} ± {s.std():.3f}")
 
labels = KMeans(2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))
agree = max((labels == y).mean(), (labels != y).mean())
print(f"K-Means (no labels!) agreement with true diagnosis = {agree:.3f}")
print(f"Adjusted Rand index = {adjusted_rand_score(y, labels):.3f}")
