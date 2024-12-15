import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

models = (
    svm.SVC(kernel="linear", C=1.0),
    svm.SVC(kernel="rbf", gamma=1.0, C=1.0),
    svm.SVC(kernel="poly", degree=3, C=1.0, gamma='auto'),
)

models = [clf.fit(X, y) for clf in models]

titles = (
    "SVM with Linear Kernel",
    "SVM with RBF Kernel",
    "SVM with Polynomial Kernel",
)

fig, sub = plt.subplots(1, 3, figsize=(15, 5))
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
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="k",
        label="Support Vectors",
    )

    scatter = ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)

    ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}")
    ax.legend()

plt.show()

