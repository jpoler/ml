import numpy as np

from linear_regression import PolynomialBasisLeastSquaresRegression
from grid.hyperparameters import (ParameterSpace, expand_grid,
                                  full_data_slices)
from data.sin import SinData

def test_expand_grid(sin_data: SinData) -> None:
    parameter_spaces = [
        ParameterSpace(
            model=PolynomialBasisLeastSquaresRegression,
            keyword="regularization_coefficient",
            base_parameters={"m_degrees": 10},
            space=np.linspace(0.0, 100.0, 10),
            base_data=sin_data,
            data_slices=full_data_slices(len(sin_data.x_train)),
        ),
        ParameterSpace(
            model=PolynomialBasisLeastSquaresRegression,
            keyword="m_degrees",
            base_parameters={},
            space=[i for i in range(10)],
            base_data=sin_data,
            data_slices=full_data_slices(len(sin_data.x_train)),
        ),
    ]

    grid = expand_grid(parameter_spaces)
    assert len(grid) == 2
    assert len(grid[0]) == 10
    assert len(grid[1]) == 10
    for c, col in enumerate(grid):
        for cell in col:
            assert cell.parameter_space.model == parameter_spaces[c].model
            assert cell.parameter_space.keyword == parameter_spaces[c].keyword
            assert cell.parameter_space.base_parameters == parameter_spaces[c].base_parameters
