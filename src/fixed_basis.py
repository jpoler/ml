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
        return np.stack(list(np.array(list(np.power(xi, m) for m in range(self.M))) for xi in x))
