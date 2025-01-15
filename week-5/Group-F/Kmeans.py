import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Sepal Length
# Sepal Width
# Petal Length
# Petal Width

# Iris setosa  0
# Iris versicolor  1
# Iris virginica   2


class CustomKMeans(KMeans):
    def __init__(self, n_clusters=3, metric='euclidean', **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)
        self.metric = metric

    def compute_distances(self, X, centers):
        if self.metric == 'cosine':
            return cosine_distances(X, centers)
        elif self.metric == 'manhattan':
            return manhattan_distances(X, centers)
        else:
            return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)

    def fit(self, X):
        self.labels = np.zeros(X.shape[0], dtype=int) # set all data point labels to zero [0,0,0]

        # define cluster center randomly
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers = X[rng.choice(X.shape[0], self.n_clusters)]

        # calculate distance and update the
        for _ in range(self.max_iter):
            distances = self.compute_distances(X, self.cluster_centers)
            # distances = np.array([
            #     [1.5, 2.3, 0.8],  # Distances for point 1 to cluster 0, cluster 1, cluster 2
            #     [2.0, 1.0, 1.5],  # Distances for point 2 to cluster 0, cluster 1, cluster 2
            #     [3.2, 2.8, 1.0],  # Distances for point 3 to cluster 0, cluster 1, cluster 2
            #     [0.5, 1.8, 1.2]  # Distances for point 4 to cluster 0, cluster 1, cluster 2
            # ])
            new_labels = np.argmin(distances, axis=1)
            # new_labels = [2, 1, 2, 0]

            if np.all(self.labels == new_labels):
                break

            self.labels = new_labels

            for cluster_id in range(self.n_clusters):
                cluster_points = X[self.labels == cluster_id]

                # prevent clusters from empty
                if cluster_points.shape[0] == 0:
                    self.cluster_centers[cluster_id] = X[rng.choice(X.shape[0], 1)]
                    continue

                if self.metric == 'manhattan':
                    self.cluster_centers[cluster_id] = np.median(cluster_points, axis=0)
                else:
                    self.cluster_centers[cluster_id] = cluster_points.mean(axis=0)

        return self

    # just give as the distance between each data point and the center
    def predict(self, X):
        distances = self.compute_distances(X, self.cluster_centers)
        return np.argmin(distances, axis=1) # return the array of the closest cluster (predicted cluster)

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2) # transform data points in 4D to 2D
X_pca = pca.fit_transform(X)

metrics = ['euclidean', 'cosine', 'manhattan']
results = []

for metric in metrics:
    kmeans = CustomKMeans(n_clusters=3, metric=metric, random_state=42, max_iter=300)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    mapping = {}
    for cluster_id in np.unique(y_pred): # just one time 0,1,2
        cluster_mask = (y_pred == cluster_id) #[True, False, True, False, False, False, True].
        counts = np.bincount(y[cluster_mask])
        majority_class = np.argmax(counts)
        mapping[cluster_id] = majority_class

    y_pred_aligned = np.array([mapping[c] for c in y_pred])

    purity = purity_score(y, y_pred)

    centers_pca = pca.transform(kmeans.cluster_centers) # transform cluster center in 4D to 2D
    results.append((metric, y_pred_aligned, centers_pca, purity))

colors = ListedColormap(['blue','orange','green'])
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=colors, edgecolor='k', vmin=0, vmax=2)
axes[0].set_title('True Labels')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')

for i, (metric, y_pred_aligned, centers_pca, purity) in enumerate(results, start=1):
    sc = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_aligned, cmap=colors, edgecolor='k', vmin=0, vmax=2)
    axes[i].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Cluster Centers')

    handles, labels = axes[i].get_legend_handles_labels()
    for idx, (cx, cy) in enumerate(centers_pca):
        handles.append(axes[i].plot([], [], ' ', label=f'C{idx}: ({cx:.2f}, {cy:.2f})')[0])

    axes[i].legend(handles=handles, loc='best')

    axes[i].set_title(f'KMeans with {metric.title()} Distance\nPurity: {purity*100:.2f}%')
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()