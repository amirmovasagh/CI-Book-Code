# Example B.2: The ROC curve, the area under the curve (AUC), and choosing a threshold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
 
X, y = load_breast_cancer(return_X_y=True)
y = 1 - y                                  # make 1 = malignant (the "positive" disease class)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000)).fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
 
fpr, tpr, thr = roc_curve(y_te, p)
print("AUC =", round(roc_auc_score(y_te, p), 4))
for t in (0.5, 0.3, 0.1):
    pred = (p >= t).astype(int)
    spec = np.mean(pred[y_te == 0] == 0)
    print(f"threshold {t}: sensitivity = {recall_score(y_te, pred):.3f}, specificity = {spec:.3f}")
 
plt.figure(figsize=(5, 4.6))
plt.plot(fpr, tpr, lw=2.5, label=f"logistic regression (AUC = {roc_auc_score(y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random guess (AUC = 0.5)")
plt.xlabel("False positive rate (1 - specificity)"); plt.ylabel("True positive rate (sensitivity)")
plt.legend(loc="lower right", fontsize=8); plt.tight_layout(); plt.savefig("figB_1.png", dpi=300)
