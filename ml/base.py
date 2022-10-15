from abc import ABC, abstractmethod
import numpy as np


class Estimator(ABC):
    def __init__(self, random_state: int):
        np.random.seed(random_state)

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
