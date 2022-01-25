import math
import numpy as np
import numpy.typing as npt

from data.sin import SinData
from linear_regression import PolynomialBasisLeastSquaresRegression

# TODO figure out whether it is proper to use biased (n) or unbiased (n-1) variance under the name "variance"
def variance(x: npt.NDArray[np.float64]) -> float:
    mean = x.mean()
    centered = x - mean
    squared = np.square(centered)
    total: float = np.sum(squared)
    return total / float(len(x))

def test_no_variance() -> None:
    epsilon = 1e-5
    zeros = np.zeros(100)
    assert math.isclose(variance(zeros), 0.0, rel_tol=epsilon), "variance should be 0 when the data is all zeros"

def test_known_variance() -> None:
    epsilon = 1e-5
    data = np.concatenate((np.full(50, 1.), np.full(50, -1.))) # type: ignore
    assert math.isclose(variance(data), 1.0, rel_tol=epsilon), "variance should be 1 when the data is all 1s and -1s"



def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    difference_variance = variance(y_pred - y_true)
    true_variance = variance(y_true)
    return 1 - (difference_variance / true_variance)

def test_polynomial_basis_least_squares_regression_explained_variance(sin_data: SinData) -> None:
    print(sin_data)
    model = PolynomialBasisLeastSquaresRegression(m_degrees=10)
    model.fit(sin_data.x_train, sin_data.y_train)
    y_pred = model.predict(sin_data.x_test)
    y_true = np.sin(sin_data.x_test)

    print(explained_variance(y_pred, y_true))

    assert False
