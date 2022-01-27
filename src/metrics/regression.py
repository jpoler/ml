import numpy as np
import numpy.typing as npt

def explained_variance(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    difference_variance: float = np.var(y_pred - y_true)
    true_variance: float = np.var(y_true)
    if not true_variance:
        raise ValueError("y_true has 0 variance")
    return 1 - (difference_variance / true_variance)
