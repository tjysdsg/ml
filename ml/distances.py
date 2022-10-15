import numpy as np
from abc import ABC, abstractmethod


class Distance(ABC):
    """
    Base class for (batched) distance measures
    """

    def __call__(self, a: np.ndarray, b: np.ndarray):
        assert a.shape[-1] == b.shape[-1]
        return self.calculate(a, b)

    @abstractmethod
    def calculate(self, a: np.ndarray, b: np.ndarray):
        pass


class EuclideanSq(Distance):
    def calculate(self, a: np.ndarray, b: np.ndarray):
        return np.sum((a - b) ** 2, axis=-1)
