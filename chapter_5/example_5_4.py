# Example 5.4: KNN with scikit-learn and choosing the best k with validation data
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
X, y = load_breast_cancer(return_X_y=True)
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=1)
 
ks, tr_acc, val_acc = range(1, 31), [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X_tr, y_tr)
    tr_acc.append(m.score(X_tr, y_tr)); val_acc.append(m.score(X_val, y_val))
 
best_k = ks[max(range(len(ks)), key=lambda i: val_acc[i])]
final = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_tmp, y_tmp)
print("k=1  -> train acc", round(tr_acc[0], 3), "| val acc", round(val_acc[0], 3))
print("Best k on validation:", best_k, "| val acc", round(max(val_acc), 3))
print("Test accuracy with best k:", round(final.score(X_te, y_te), 3))
 
plt.figure(figsize=(7, 3.8))
plt.plot(ks, tr_acc, "o-", ms=3, label="train"); plt.plot(ks, val_acc, "o-", ms=3, label="validation")
plt.axvline(best_k, ls="--", c="k"); plt.xlabel("k (number of neighbors)"); plt.ylabel("accuracy")
plt.legend(); plt.tight_layout(); plt.savefig("fig5_1.png", dpi=300)
