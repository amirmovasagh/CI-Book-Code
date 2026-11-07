# Example 2.10: A multilayer neural network with scikit-learn for breast cancer diagnosis
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
 
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
 
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(32, 16),   # two hidden layers
                  activation="relu", solver="adam",
                  learning_rate_init=0.001, max_iter=500,
                  early_stopping=True, random_state=0))
model.fit(X_tr, y_tr)
 
mlp = model[-1]
print("Layers (weights shapes):", [w.shape for w in mlp.coefs_])
print("Epochs run:", mlp.n_iter_)
print("Test accuracy:", round(model.score(X_te, y_te), 3))
print("Confusion matrix:\n", confusion_matrix(y_te, model.predict(X_te)))
print(classification_report(y_te, model.predict(X_te),
                            target_names=["malignant", "benign"], digits=3))
