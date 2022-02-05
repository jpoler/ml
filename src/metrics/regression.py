import numpy as np
import numpy.typing as npt

def check_sizes(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    if x.size != y.size:
        raise ValueError(f"sizes do not match {x.size} != {y.size}")
    if x.size == 0:
        raise ValueError("arrays provided are of length 0")

def r2_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    check_sizes(y_true, y_pred)
    d: float = np.square(y_true - y_true.mean()).sum()
    if not d:
        raise ValueError("y_true has 0 variance")
    n: float = np.square(y_true - y_pred).sum()
    print(f"n: {n}, d: {d}")
    return 1. - (n / d)

def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    check_sizes(y_true, y_pred)
    difference_variance: float = np.var(y_pred - y_true)
    true_variance: float = np.var(y_true)
    if not true_variance:
        raise ValueError("y_true has 0 variance")
    return 1 - (difference_variance / true_variance)

def mean_squared_error(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    check_sizes(y_true, y_pred)
    mse: float = np.square(y_true - y_pred).mean()
    return mse
