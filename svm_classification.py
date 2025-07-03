# SVM Classifier using Breast Cancer Dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("✅ Script started.")

# Step 1: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"\n🔹 Dataset shape: {X.shape}")
print("🔹 Target labels:", np.unique(y))

# Step 2: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train SVM with linear kernel
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

print("\n🔸 Linear SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_linear))

# Step 5: Train SVM with RBF kernel
rbf_svm = SVC(kernel='rbf', gamma='scale', C=1.0)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print("\n🔸 RBF SVM Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))

# Step 6: Cross-validation on RBF
cv_scores = cross_val_score(rbf_svm, X_scaled, y, cv=5)
print("\n🔍 Cross-validation scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))

# Step 7: Visualize decision boundary (using PCA to reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

svm_vis = SVC(kernel='rbf', gamma='scale', C=1.0)
svm_vis.fit(X_train_pca, y_train_pca)

# Plot decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.grid(True)
    plt.savefig("svm_decision_boundary.png")
    plt.show()

plot_decision_boundary(svm_vis, X_test_pca, y_test_pca, "SVM with RBF Kernel (PCA 2D)")

print("\n📊 Decision boundary saved as 'svm_decision_boundary.png'")
input("\n✅ Script finished. Press Enter to exit...")
