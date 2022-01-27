import math
import numpy as np
import pytest

from metrics.regression import explained_variance

def test_explained_variance_zero_true_variance() -> None:
    zeros = np.zeros(100)
    with pytest.raises(ValueError):
        explained_variance(zeros, zeros)

def test_explained_variance_matching_data() -> None:
    epsilon = 1e-5
    x = np.array([0.0,1.0])
    assert math.isclose(explained_variance(x, x), 1.0, rel_tol=epsilon), "matching data should have explained variance near 1.0"
