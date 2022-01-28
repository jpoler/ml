from abc import abstractmethod
import numpy as np
import numpy.typing as npt
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
        phi = np.empty((len(x), self.M))
        for index, _ in np.ndenumerate(phi):
            phi[index] = x[index[0]]**index[1]
        return phi
