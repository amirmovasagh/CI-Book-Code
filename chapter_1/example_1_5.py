# Example 1.5: Dimensionality reduction of breast cancer data with PCA and LDA
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
data = load_breast_cancer()
X, y = data.data, data.target            # 569 samples, 30 features
print("Original shape:", X.shape)
 
X_std = StandardScaler().fit_transform(X)   # feature scaling
 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
print("Total kept:", round(pca.explained_variance_ratio_.sum(), 3))
 
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_std, y)       # LDA uses the labels y
print("LDA shape:", X_lda.shape)
 
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for cls, color, name in [(0, "#D1495B", "Malignant"), (1, "#3C6E8F", "Benign")]:
    ax[0].scatter(X_pca[y == cls, 0], X_pca[y == cls, 1], s=10, c=color, label=name)
    ax[1].hist(X_lda[y == cls, 0], bins=30, alpha=0.7, color=color, label=name)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_title("PCA: 30D -> 2D")
ax[1].set_xlabel("LD1"); ax[1].set_title("LDA: 30D -> 1D")
ax[0].legend(); ax[1].legend()
plt.tight_layout(); plt.savefig("fig1_4.png", dpi=300)
