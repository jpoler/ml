import math
import numpy as np
import numpy.typing as npt

from metrics.variance import variance, explained_variance

def test_no_variance() -> None:
    epsilon = 1e-5
    zeros = np.zeros(100)
    assert math.isclose(variance(zeros), 0.0, rel_tol=epsilon), "variance should be 0 when the data is all zeros"

def test_known_variance() -> None:
    epsilon = 1e-5
    data = np.concatenate((np.full(50, 1.), np.full(50, -1.))) # type: ignore
    assert math.isclose(variance(data), 1.0, rel_tol=epsilon), "variance should be 1 when the data is all 1s and -1s"
