from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Model(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass
