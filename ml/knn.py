from ml.base import Estimator
from ml.distances import EuclideanSq
import numpy as np
from typing import Literal


class KNN(Estimator):
    def __init__(self, k: int, distance=EuclideanSq(), reduce: Literal['vote', 'mean'] = 'vote'):
        super().__init__()
        self.k = k
        self.distance = distance
        self.x = None
        self.y = None
        self.reduce = reduce

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(func1d=self.predict_one, axis=1, arr=x)

    def predict_one(self, x: np.ndarray):
        distances = self.distance(self.x, x)
        idx = np.argsort(distances)[:self.k]

        if self.reduce == 'vote':
            return np.bincount(self.y[idx]).argmax()
        elif self.reduce == 'mean':
            return np.mean(self.y[idx])


def test_classifier():
    from sklearn.neighbors import KNeighborsClassifier

    x = np.random.rand(50, 16)
    y = np.random.randint(0, 5, 50)
    knn = KNN(k=4, reduce='vote')
    standard = KNeighborsClassifier(n_neighbors=4)

    knn.fit(x, y)
    standard.fit(x, y)

    x_test = np.random.rand(10, 16)
    assert np.allclose(knn.predict(x_test), standard.predict(x_test))


def test_regressor():
    from sklearn.neighbors import KNeighborsRegressor

    x = np.random.rand(50, 16)
    y = np.random.rand(50)
    knn = KNN(k=4, reduce='mean')
    standard = KNeighborsRegressor(n_neighbors=4)

    knn.fit(x, y)
    standard.fit(x, y)

    x_test = np.random.rand(10, 16)
    assert np.allclose(knn.predict(x_test), standard.predict(x_test))
