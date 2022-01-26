import numpy as np
import numpy.typing as npt

# TODO figure out whether it is proper to use biased (n) or unbiased (n-1) variance under the name "variance"
def variance(x: npt.NDArray[np.float64]) -> float:
    mean = x.mean()
    centered = x - mean
    squared = np.square(centered)
    total: float = np.sum(squared)
    return total / float(len(x))

def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    difference_variance = variance(y_pred - y_true)
    true_variance = variance(y_true)
    return 1 - (difference_variance / true_variance)
