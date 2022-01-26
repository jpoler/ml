import numpy as np
from fixed_basis import PolynomialBasisMixin
from typing import Any

class PolynomialBasisTest(PolynomialBasisMixin):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

def test_polynomial_basis_mixin() -> None:
    basis_class = PolynomialBasisTest(m_degrees=5)
    x = np.arange(10, dtype=np.float64)
    phi = basis_class.phi(x)
    assert phi.shape == (10, 5)

    for index, x in np.ndenumerate(phi):
        print(f"index: {index}")
        assert x == float(index[0])**index[1]
