from dataclasses import dataclass
import numpy as np
from sklearn import datasets # type: ignore

from data.data import Data

@dataclass
class IrisData(Data):
    pass

def iris_data(n_train_proportion: float=.8) -> IrisData:
    X, y = datasets.load_iris(return_X_y=True)
    n = len(y)
    n_train = int(n_train_proportion * n)
    y = y.reshape((len(y), 1))
    Y = np.zeros((len(y), 3))
    idx_y0 = y == 0
    idx_y1 = y == 1
    idx_y2 = y == 2
    idx = np.hstack([idx_y0, idx_y1, idx_y2])
    Y[idx] = 1.

    p = np.random.default_rng().permutation(n)
    X_p = X[p, :]
    Y_p = Y[p, :]

    return IrisData(x_train=X_p[:n_train, :], y_train=Y_p[:n_train, :], x_test=X_p[n_train:, :], y_test=Y_p[n_train:, :])

