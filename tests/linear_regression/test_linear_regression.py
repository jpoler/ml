import numpy as np
import pytest

from data.sin import SinData
from grid import hyperparameters
from metrics.regression import explained_variance, mean_squared_error, r2_score
from linear_regression import GaussianBasisLeastSquaresRegression, PolynomialBasisLeastSquaresRegression, PolynomialBasisBayesianLinearRegression

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

def test_polynomial_basis_least_squares_regression_baseline_r2_score(sin_data: SinData) -> None:
    scores = []
    for i in (100,):
        x_train, y_train = sin_data.x_train[:i], sin_data.y_train[:i]
        x_test, y_test = sin_data.x_test, sin_data.y_test
        model = PolynomialBasisLeastSquaresRegression(m_degrees=10)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)
        print(f"W: {model.W}")
        print(f"W length: {np.linalg.norm(model.W)}") # type: ignore

    print(f"least squares errors {scores}")
    assert all(e > 0.5 for e in scores)

@pytest.mark.skip(reason="this test flaps, indicating that it's not a good test, rewrite")
def test_gaussian_basis_least_squares_regression_mean_squared_error_decreases_with_more_data(sin_data: SinData) -> None:
    errors = []
    num_basis = 10
    xmin, xmax = np.amin(sin_data.x_train), np.amax(sin_data.y_train)
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

@pytest.mark.parametrize("sin_data", [dict(noise_stddev=.1)], indirect=True)
def test_polynomial_basis_bayesian_linear_regression_baseline_r2_score(sin_data: SinData) -> None:
    scores = []
    num_basis = 10
    for i in (30,):
        x_train, y_train = sin_data.x_train[:i], sin_data.y_train[:i]
        x_test, y_test = sin_data.x_test, sin_data.y_test
        model = PolynomialBasisBayesianLinearRegression(m_degrees=num_basis, max_evidence_iterations=100)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)

    print(f"r2_scores {scores}")
    assert all([e > 0.5 for e in scores])

def test_polynomial_basis_bayesian_linear_regression_precision_strictly_increases(sin_data: SinData) -> None:
    precision_norms = []
    num_basis = 10
    # first train the model with 100 data points to ensure convergence
    model = PolynomialBasisBayesianLinearRegression(alpha=1., m_degrees=num_basis, max_evidence_iterations=100)
    model.fit(sin_data.x_train[:30], sin_data.y_train[:30])
    norm: float = np.linalg.norm(model.w_precision) # type: ignore
    precision_norms.append(norm)
    for data in hyperparameters.data_generator(sin_data, hyperparameters.single_slices(30, 40)):
        model.fit(data.x_train, data.y_train)
        norm: float = np.linalg.norm(model.w_precision) # type: ignore
        precision_norms.append(norm)
    print(f"precision_norms: {precision_norms}")
    assert sorted(precision_norms) == precision_norms

def test_polynomial_basis_bayesian_linear_regression_gamma_strictly_increases(sin_data: SinData) -> None:
    gammas = []
    num_basis = 10
    # first train the model with 100 data points to ensure convergence
    model = PolynomialBasisBayesianLinearRegression(alpha=1., m_degrees=num_basis, max_evidence_iterations=100)
    model.fit(sin_data.x_train[:30], sin_data.y_train[:30])
    gamma = model.gamma(model.alpha, model.beta*model.phi_t_phi)
    gammas.append(gamma)
    for data in hyperparameters.data_generator(sin_data, hyperparameters.single_slices(30, 40)):
        model.fit(data.x_train, data.y_train)
        gamma = model.gamma(model.alpha, model.beta*model.phi_t_phi)
        gammas.append(gamma)
    print(f"gammas: {gammas}")
    assert sorted(gammas) == gammas
