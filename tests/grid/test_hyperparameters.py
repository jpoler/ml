from itertools import islice
from typing import List, Tuple

from grid.hyperparameters import (ParameterSpace, data_generator, expand_grid,
                                  full_data_slices, increasing_subslices, mean_squared_error_metric, parameter_generator)
from data.sin import SinData


def test_parameter_generator() -> None:
    space = list([i for i in range(10)])
    base_parameters = {"foo": 42}
    for i, parameters in enumerate(parameter_generator(space, "bar", base_parameters)):
        assert parameters["foo"] == 42
        assert parameters["bar"] == space[i]

def test_increasing_subslices(sin_data: SinData) -> None:
    low, high, n = (11, 21, 10)
    slices = increasing_subslices(low, high, n)
    for i, data in enumerate(data_generator(sin_data, slices)):
        j = i
        assert data.x_train[0] == sin_data.x_train[low]
        assert data.x_train[-1] == sin_data.x_train[low+i]

        assert data.y_train[0] == sin_data.y_train[low]
        assert data.y_train[-1] == sin_data.y_train[low+i]

        assert data.x_test[0] == sin_data.x_test[low]
        assert data.x_test[-1] == sin_data.x_test[low+i]

        assert data.y_test[0] == sin_data.y_test[low]
        assert data.y_test[-1] == sin_data.y_test[low+i]

    assert low + j == high - 1

def test_full_data_slices(sin_data: SinData) -> None:
    n = 20
    slices = full_data_slices(n)
    for data in islice(data_generator(sin_data, slices), 10):
        len(data.x_train) == n
        len(data.y_train) == n
        len(data.x_test) == n
        len(data.y_test) == n

def test_expand_grid(sin_data_and_parameter_spaces: Tuple[SinData, List[ParameterSpace]]) -> None:
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

def test_mean_squared_error_metric(sin_data_and_parameter_spaces: Tuple[SinData, List[ParameterSpace]]) -> None:
    sin_data, parameter_spaces = sin_data_and_parameter_spaces
    grid = expand_grid(parameter_spaces)
    mse_grid = mean_squared_error_metric(grid)
    assert len(grid) == len(mse_grid)
    for r, row in enumerate(mse_grid):
        assert len(row) == len(grid[r])
        assert all(row)
