from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy.stats import norm # type: ignore
from typing import Any

class FixedBasisFunctionMixin(ABC):
    @abstractmethod
    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

    @property
    @abstractmethod
    def basis_dimensionality(self) -> int:
        pass


class PolynomialBasisMixin(FixedBasisFunctionMixin):
    def __init__(self, m_degrees: int = 10, *args: Any, **kwargs: Any) -> None:
        self.M = m_degrees
        super().__init__(*args, **kwargs)

    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = np.empty((len(x), self.M), dtype=np.float64)
        for index, _ in np.ndenumerate(phi):
            phi[index] = x[index[0]]**index[1]
        return phi

    @property
    def basis_dimensionality(self) -> int:
        return self.M

class GaussianBasisMixin(FixedBasisFunctionMixin):
    # todo replace num with "dim" or "dimensionality"
    def __init__(self, low: int = 0, high: int = 1, num: int = 2, stddev: float = 1., *args: Any, **kwargs: Any):
        self.low, self.high, self.num, self.stddev = low, high, num, stddev
        self.means = np.linspace(self.low, self.high, self.num)
        super().__init__(*args, **kwargs)

    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = np.empty((len(x), self.num), dtype=np.float64)
        for index, _ in np.ndenumerate(phi):
            phi[index] = norm.pdf(x[index[0]], self.means[index[1]], self.stddev)
        return phi

    @property
    def basis_dimensionality(self) -> int:
        return self.num


