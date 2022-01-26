from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Any, cast, Optional

from model import Model
from fixed_basis import PolynomialBasisMixin

class PolynomialBasisLeastSquaresRegression(PolynomialBasisMixin,
                             Model[npt.NDArray[np.float64]]):
    def __init__(self, regularization_coefficient: Optional[float] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = regularization_coefficient

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        phi = self.phi(X)
        if self.regularization_coefficient:
            inner = self.regularization_coefficient * np.eye(phi.T.shape[0]) + phi.T @ phi # type: ignore
        else:
            inner = phi.T @ phi
        inv = np.linalg.inv(inner) # type: ignore
        self.W: npt.NDArray[np.float64] = inv @ phi.T @ y

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = self.phi(x)
        return self.W @ phi.T
