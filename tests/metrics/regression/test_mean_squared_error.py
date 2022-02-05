import math
import numpy as np
import pytest

from metrics.regression import mean_squared_error
from constants import test_epsilon

def test_mean_squared_error_known_variance() -> None:
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    mse = mean_squared_error(y_true, y_pred)
    assert math.isclose(mse, 1.0, rel_tol=test_epsilon)

def test_mean_squared_error_positive() -> None:
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    assert mean_squared_error(y_true, y_pred) > 0
    assert mean_squared_error(y_pred, y_true) > 0

def test_mean_squared_error_length_zero() -> None:
    with pytest.raises(ValueError):
        mean_squared_error(np.array([]), np.array([]))

def test_mean_squared_error_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1]), np.array([]))
