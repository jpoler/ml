import math
import numpy as np
import pytest

from metrics.regression import r2_score
from constants import test_epsilon

def test_r2_score_zero_true_variance() -> None:
    zeros = np.zeros(100)
    with pytest.raises(ValueError):
        r2_score(zeros, zeros)

def test_r2_score_matching_data() -> None:
    x = np.array([0.0,1.0])
    assert math.isclose(r2_score(x, x), 1.0, rel_tol=test_epsilon), "matching data should have r2 score near 1.0"
