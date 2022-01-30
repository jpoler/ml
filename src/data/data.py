from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class Data:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]
