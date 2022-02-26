from math import factorial
import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm # type: ignore
from typing import Any

from fixed_basis import GaussianBasisMixin, PolynomialBasisMixin

class PolynomialBasisTest(PolynomialBasisMixin):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class GaussianBasisTest(GaussianBasisMixin):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize("x",
                         [
                             np.arange(10, dtype=np.float64),
                             np.arange(100, dtype=np.float64).reshape((10, 10)),
                         ])
def test_polynomial_basis_mixin(x: npt.NDArray[np.float64]) -> None:
    m = 3
    basis_class = PolynomialBasisTest(m_degrees=m)
    phi = basis_class.phi(x)
    if len(x.shape) == 1:
        x = x.reshape((len(x), 1))
    width = len(x[0,:])
    phi_width = 0
    for r in range(basis_class.basis_dimensionality):
        phi_width += int(factorial(width+r-1) / factorial(r) / factorial(width-1))
    assert phi.shape == (x.shape[0], phi_width)


def test_gaussian_basis_mixin() -> None:
    stddev = 1.
    basis_class = GaussianBasisTest(low=0, high=10, num=11, stddev=stddev)
    x = np.arange(10, dtype=np.float64)
    phi = basis_class.phi(x)
    assert phi.shape == (10, 11)

    v: float
    for index, v in np.ndenumerate(phi):
        assert v == norm.pdf(x[index[0]], index[1], stddev)
