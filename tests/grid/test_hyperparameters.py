from itertools import islice
import numpy as np
from typing import List, Tuple

from data.sin import SinData
from grid.hyperparameters import (ParameterSpace, data_generator, expand_grid,
                                  full_data_slices, increasing_subslices, mean_squared_error_metric, parameter_generator)
from linear_regression import PolynomialBasisLeastSquaresRegression


def test_parameter_generator() -> None:
    space = list([i for i in range(10)])
    base_parameters = {"foo": 42}
    for i, parameters in enumerate(parameter_generator(space, "bar", base_parameters)):
        assert parameters["foo"] == 42
        assert parameters["bar"] == space[i]

import pytest
@pytest.mark.focus
@pytest.mark.parametrize("arg_tuple", [(11, 21, 10), (50, 150, 10)])
def test_increasing_subslices(sin_data: SinData, arg_tuple: Tuple[int, int, int]) -> None:
    low, high, n = arg_tuple
    k = (high-low) // n
    slices = increasing_subslices(low, high, n)
    slice_count = 0
    for i, data in enumerate(data_generator(sin_data, slices)):
        slice_count += 1
        print(low, low+(i+1)*k)
        assert np.array_equal(data.x_train[0:(i+1)*k], sin_data.x_train[low:low+(i+1)*k])
        assert np.array_equal(data.y_train[0:(i+1)*k], sin_data.y_train[low:low+(i+1)*k])
        assert np.array_equal(data.x_test[0:(i+1)*k], sin_data.x_test[low:low+(i+1)*k])
        assert np.array_equal(data.y_test[0:(i+1)*k], sin_data.y_test[low:low+(i+1)*k])
    assert slice_count == n

def test_full_data_slices(sin_data: SinData) -> None:
    n = 20
    slices = full_data_slices(n)
    for data in islice(data_generator(sin_data, slices), 10):
        len(data.x_train) == n
        len(data.y_train) == n
        len(data.x_test) == n
        len(data.y_test) == n

def test_expand_grid(sin_data_and_parameter_spaces: Tuple[SinData, List[ParameterSpace[PolynomialBasisLeastSquaresRegression]]]) -> None:
    sin_data, parameter_spaces = sin_data_and_parameter_spaces
    grid = expand_grid(parameter_spaces)
    assert len(grid) == 2
    assert len(grid[0]) == 10
    assert len(grid[1]) == 10
    for r, row in enumerate(grid):
        for cell in row:
            assert cell.parameter_space.model == parameter_spaces[r].model
            assert cell.parameter_space.keyword == parameter_spaces[r].keyword
            assert cell.parameter_space.base_parameters == parameter_spaces[r].base_parameters
            assert cell.predictions is not None

def test_mean_squared_error_metric(sin_data_and_parameter_spaces: Tuple[SinData, List[ParameterSpace[PolynomialBasisLeastSquaresRegression]]]) -> None:
    sin_data, parameter_spaces = sin_data_and_parameter_spaces
    grid = expand_grid(parameter_spaces)
    mse_grid = mean_squared_error_metric(grid)
    assert len(grid) == len(mse_grid)
    for r, row in enumerate(mse_grid):
        assert len(row) == len(grid[r])
        assert all(row)
