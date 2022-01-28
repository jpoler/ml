from abc import abstractmethod
import numpy as np
import numpy.typing as npt
from scipy.stats import norm # type: ignore
from typing import Any, TypeVar, Generic

T = TypeVar("T")

class FixedBasisFunctionMixin(Generic[T]):
    @abstractmethod
    def phi(self, x: T) -> T:
        pass

class PolynomialBasisMixin(FixedBasisFunctionMixin[npt.NDArray[np.float64]]):
    def __init__(self, m_degrees: int = 10, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.M = m_degrees

    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = np.empty((len(x), self.M), dtype=np.float64)
        for index, _ in np.ndenumerate(phi):
            phi[index] = x[index[0]]**index[1]
        return phi

class GaussianBasisMixin(FixedBasisFunctionMixin[npt.NDArray[np.float64]]):
    def __init__(self, low: int = 0, high: int = 1, num: int = 2, stddev: float = 1., *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.low, self.high, self.num, self.stddev = low, high, num, stddev
        self.means = np.linspace(self.low, self.high, self.num)

    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        print(f"stddev: {self.stddev}")
        phi = np.empty((len(x), self.num), dtype=np.float64)
        for index, _ in np.ndenumerate(phi):
            phi[index] = norm.pdf(x[index[0]], self.means[index[1]], self.stddev)
        return phi



