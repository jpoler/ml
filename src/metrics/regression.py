import numpy as np
import numpy.typing as npt

def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    difference_variance: float = np.var(y_pred - y_true)
    true_variance: float = np.var(y_true)
    if not true_variance:
        raise ValueError("y_true has 0 variance")
    return 1 - (difference_variance / true_variance)

def mean_squared_error(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true size {y_true.size} does not match y_pred size {y_pred.size}")
    if y_true.size == 0:
        raise ValueError("arrays provided are of length 0")
    mse: float = np.square(y_true - y_pred).mean()
    return mse
