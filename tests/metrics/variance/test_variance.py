import math
import numpy as np
import pytest

from metrics.variance import variance, explained_variance

def test_empty_array() -> None:
    with pytest.raises(ValueError):
        variance(np.array([]))

def test_no_variance() -> None:
    epsilon = 1e-5
    zeros = np.zeros(100)
    assert math.isclose(variance(zeros), 0.0, rel_tol=epsilon), "variance should be 0 when the data is all zeros"

def test_known_variance() -> None:
    epsilon = 1e-5
    data = np.concatenate((np.full(50, 1.), np.full(50, -1.))) # type: ignore
    assert math.isclose(variance(data), 1.0, rel_tol=epsilon), "variance should be 1 when the data is all 1s and -1s"

def test_explained_variance_zero_true_variance() -> None:
    zeros = np.zeros(100)
    with pytest.raises(ValueError):
        explained_variance(zeros, zeros)

def test_explained_variance_matching_data() -> None:
    epsilon = 1e-5
    x = np.array([0.0,1.0])
    assert math.isclose(explained_variance(x, x), 1.0, rel_tol=epsilon), "matching data should have explained variance near 1.0"
