import numpy as np
from fixed_basis import GaussianBasisMixin, PolynomialBasisMixin
from scipy.stats import norm # type: ignore
from typing import Any

class PolynomialBasisTest(PolynomialBasisMixin):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


class GaussianBasisTest(GaussianBasisMixin):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


def test_polynomial_basis_mixin() -> None:
    basis_class = PolynomialBasisTest(m_degrees=5)
    x = np.arange(10, dtype=np.float64)
    phi = basis_class.phi(x)
    assert phi.shape == (10, 5)

    v: float
    for index, v in np.ndenumerate(phi):
        assert v == float(index[0])**index[1]


def test_gaussian_basis_mixin() -> None:
    stddev = 1.
    basis_class = GaussianBasisTest(low=0, high=10, num=11, stddev=stddev)
    x = np.arange(10, dtype=np.float64)
    phi = basis_class.phi(x)
    assert phi.shape == (10, 11)

    v: float
    for index, v in np.ndenumerate(phi):
        assert v == norm.pdf(x[index[0]], index[1], stddev)
