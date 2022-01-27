import numpy as np

from data.sin import SinData
from metrics.regression import explained_variance
from linear_regression import PolynomialBasisLeastSquaresRegression

def test_polynomial_basis_least_squares_regression_explained_variance(sin_data: SinData) -> None:
    print(sin_data)
    model = PolynomialBasisLeastSquaresRegression(m_degrees=10)
    model.fit(sin_data.x_train, sin_data.y_train)
    y_pred = model.predict(sin_data.x_test)
    y_true = np.sin(sin_data.x_test)

    print(explained_variance(y_pred, sin_data.y_test))
    print(explained_variance(y_pred, y_true))

    assert False
