"""General class for metrics for which the reduction error is computed."""
from abc import ABC, abstractmethod
import numpy as np


class MorErrorMetric(ABC):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
    
    @abstractmethod
    def eval(self, data, target) -> dict:
        ...


class RelEuclideanMorErrorMetric(MorErrorMetric):
    def __init__(self, sqrt: bool = True, indices: np.ndarray = None, name: str = '2norm') -> None:
        super().__init__(name)
        self.sqrt = sqrt
        self.indices = indices

    def eval(self, data, target):
        if not self.indices is None:
            data = data[..., self.indices]
            target = target[..., self.indices]
        err = np.sum((data - target)**2, axis=-1) / np.sum(target**2)
        if self.sqrt:
            err = np.sqrt(err)
        return {self.name: err}
