import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

binary_class_indices = (y == 0) | (y == 1)  # Select only class 0 and class 1
X = X[binary_class_indices]
y = y[binary_class_indices]

models = (
    svm.SVC(kernel="linear", C=1.0),
    svm.SVC(kernel="rbf", gamma=1.0, C=1.0),
    svm.SVC(kernel="poly", degree=3, C=1.0, gamma="auto"),
)

models = [clf.fit(X, y) for clf in models]

titles = (
    "SVM with Linear Kernel",
    "SVM with RBF Kernel",
    "SVM with Polynomial Kernel",
)

fig, sub = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub):
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )

    ax.scatter(
        X0, X1,
        c=y,
        cmap=plt.cm.coolwarm,
        s=20,
        edgecolors="face",
    )

    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        label="Support Vectors",
    )

    if clf.kernel == "linear":
        w = clf.coef_[0]
        slope = -w[0] / w[1]
        intercept = -clf.intercept_[0] / w[1]
        x_min, x_max = ax.get_xlim()
        x_values = np.linspace(x_min, x_max, 100)
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        ax.plot(x_values, slope * x_values + intercept + margin, "k--", linewidth=1 )
        ax.plot(x_values, slope * x_values + intercept - margin, "k--", linewidth=1)
    else:
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
            np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
        )
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")

    y_pred = clf.predict(X)

    accuracy = accuracy_score(y, y_pred)

    ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}")
    ax.legend()

plt.show()
