# Example 1.9: The complete roadmap of a project — from data to deployment
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 
# 1) Data collection
X, y = load_breast_cancer(return_X_y=True)
 
# 2) Data splitting (60/20/20)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=0)
 
# 3) Cleaning/preprocessing + 4) building and training several candidate models
candidates = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN (k=5)":          make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "DecisionTree":       DecisionTreeClassifier(max_depth=4, random_state=0),
}
 
# 5) Evaluate on the validation data and select the best model
scores = {}
for name, m in candidates.items():
    m.fit(X_tr, y_tr)
    scores[name] = m.score(X_val, y_val)
    print(f"{name:<20} validation accuracy = {scores[name]:.3f}")
best = max(scores, key=scores.get)
print("Best model:", best)
 
# 6) Final evaluation — only once — on the test data
final = candidates[best]
print("Test accuracy:", round(final.score(X_test, y_test), 3))
 
# 7) Deployment: save the model and reuse it
joblib.dump(final, "cancer_model.joblib")
loaded = joblib.load("cancer_model.joblib")
print("Prediction for first test sample:", loaded.predict(X_test[:1])[0],
      "| true label:", y_test[0])
