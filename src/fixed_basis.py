from abc import ABC, abstractmethod
from functools import reduce
from itertools import combinations_with_replacement
from math import factorial
import numpy as np
import numpy.typing as npt
from operator import mul
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
    def __init__(self, m_degrees: int = 2, *args: Any, **kwargs: Any) -> None:
        self.M = m_degrees
        super().__init__(*args, **kwargs)

    def phi(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if len(x.shape) == 1:
            x = x.reshape((len(x), 1))
        n = len(x)
        width = len(x[0,:])
        phi_width = 0
        for r in range(self.M):
            phi_width += int(factorial(width+r-1) / factorial(r) / factorial(width-1))
        phi = np.empty((n, phi_width), dtype=np.float64)
        for i in range(n):
            k = 0
            for j in range(self.M):
                gen = (reduce(mul, p, 1) for p in combinations_with_replacement(x[i,:], j))
                for e in gen:
                    phi[i, k] = e
                    k += 1
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


