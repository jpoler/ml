import numpy as np
import numpy.typing as npt

# TODO figure out whether it is proper to use biased (n) or unbiased (n-1) variance under the name "variance"
def variance(x: npt.NDArray[np.float64]) -> float:
    if len(x) == 0:
        raise ValueError("cannot compute variance for size 0 array")
    mean = x.mean()
    centered = x - mean
    squared = np.square(centered)
    total: float = np.sum(squared)
    return total / float(len(x))

def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    difference_variance = variance(y_pred - y_true)
    true_variance = variance(y_true)
    if not true_variance:
        raise ValueError("y_true has 0 variance")
    return 1 - (difference_variance / true_variance)
