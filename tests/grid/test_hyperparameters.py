import numpy as np

from linear_regression import PolynomialBasisLeastSquaresRegression
from grid.hyperparameters import full_data_grid, ParameterSpace
from data.sin import sin_data

def test_full_data_grid():
    models = [
        PolynomialBasisLeastSquaresRegression,
        PolynomialBasisLeastSquaresRegression,
    ]
    param_spaces = [
        ParameterSpace(keyword="regularization_coefficient", space=np.linspace(0.0, 100.0, 10)),
        ParameterSpace(keyword="m_degrees", space=[i for i in range(10)])
    ]
    data = sin_data()

    grid = full_data_grid(models, data, {"regularization_coefficient": 1., "m_degrees": 5}, param_spaces)

    print(list(grid))

    assert False
