import numpy as np
import pytest

from data.sin import SinData
from metrics.regression import explained_variance, mean_squared_error
from linear_regression import GaussianBasisLeastSquaresRegression, PolynomialBasisLeastSquaresRegression

def test_polynomial_basis_least_squares_regression_explained_variance(sin_data: SinData) -> None:
    model = PolynomialBasisLeastSquaresRegression(m_degrees=10)
    model.fit(sin_data.x_train, sin_data.y_train)
    y_pred = model.predict(sin_data.x_test)
    y_true = np.sin(sin_data.x_test)

    test_explained_variance = explained_variance(y_pred, sin_data.y_test)
    true_explained_variance = explained_variance(y_pred, y_true)

    assert test_explained_variance > 0, "the model should perform better than guessing the mean of y"
    assert true_explained_variance > 0, "the model should perform better than guessing the mean of y"
    assert true_explained_variance > test_explained_variance, "the model should predict close to the conditional mean" \
        "and thus perform better without noise"

@pytest.mark.skip(reason="this test flaps, indicating that it's not a good test, rewrite")
def test_polynomial_basis_least_squares_regression_mean_squared_error_decreases_with_more_data(sin_data: SinData) -> None:
    errors = []
    for i in (1, 10, 500):
        x_test, y_test = sin_data.x_train[:i], sin_data.y_train[:i]
        model = PolynomialBasisLeastSquaresRegression(m_degrees=10)
        model.fit(x_test, y_test)
        y_pred = model.predict(sin_data.x_test)
        y_true = np.sin(sin_data.x_test)
        mse = mean_squared_error(y_true, y_pred)
        errors.append(mse)

    assert sorted(errors, reverse=True) == errors

@pytest.mark.skip(reason="this test flaps, indicating that it's not a good test, rewrite")
def test_gaussian_basis_least_squares_regression_mean_squared_error_decreases_with_more_data(sin_data: SinData) -> None:
    errors = []
    num_basis = 10
    xmin, xmax = np.amin(sin_data.x_train), np.amin(sin_data.y_train)
    stddev = (xmax - xmin) / num_basis
    for i in (1, 10, 500):
        x_test, y_test = sin_data.x_train[:i], sin_data.y_train[:i]
        model = GaussianBasisLeastSquaresRegression(low=xmin, high=xmax, num=10, stddev=stddev)
        model.fit(x_test, y_test)
        y_pred = model.predict(sin_data.x_test)
        y_true = np.sin(sin_data.x_test)
        mse = mean_squared_error(y_true, y_pred)
        errors.append(mse)

    assert sorted(errors, reverse=True) == errors
