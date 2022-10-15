from ml.base import Estimator
from ml.knn import KNN
from ml.distances import EuclideanSq
import numpy as np


class KMeans(Estimator):
    def __init__(self, k: int, distance=EuclideanSq(), max_iter=300, tol=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.distance = distance
        self.max_iter = max_iter
        self.tol = tol
        self.knn = KNN(1, distance, 'vote', *args, **kwargs)
        self.means = None

    def fit(self, x: np.ndarray, _=None):
        feat_size = x.shape[-1]
        self.means = x[np.random.permutation(x.shape[0])[:self.k]]
        prev_means = self.means

        assignment = [[] for _ in range(self.k)]
        for i in range(self.max_iter):
            x_clusters = self.assign_clusters(x)
            for j, c in enumerate(x_clusters):
                assignment[c].append(x[j])
            self.means = np.asarray([np.mean(v, axis=0) for v in assignment])

            if np.linalg.norm(self.means - prev_means) <= self.tol:
                break

    def assign_clusters(self, x: np.ndarray):
        self.knn.fit(self.means, np.asarray([j for j in range(self.k)]))
        return self.knn.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.assign_clusters(x)


def test():
    from sklearn.cluster import KMeans as Standard

    x = np.random.rand(50, 16) * 10
    kmeans = KMeans(k=4, max_iter=100, random_state=42)
    standard = Standard(n_clusters=4, init='random', max_iter=100, random_state=42)

    kmeans.fit(x)
    standard.fit(x)

    x_test = np.random.rand(10, 16)
    assert np.allclose(kmeans.predict(x_test), standard.predict(x_test))
