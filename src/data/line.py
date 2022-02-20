from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from data.data import Data

@dataclass
class LineData(Data):
    pass

@dataclass
class LineCurve:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

def line_data(
        n_train: int = 80,
        n_test: int = 20,
        x_low: float = 0.,
        x_high: float = 1.,
        slope: float = 1.,
        intercept: float = 0.,
        noise_stddev: float = .5,
) -> LineData:
    rng = np.random.default_rng()
    n_total = n_train + n_test
    x = rng.uniform(low=x_low, high=x_high, size=n_total)
    noise = rng.normal(loc=0., scale=noise_stddev, size=n_total)
    y: npt.NDArray[np.float64] = (intercept + slope*x) + noise
    return LineData(x_train=x[:n_train], y_train=y[:n_train], x_test=x[n_train:], y_test=y[n_train:])

def line_curve(
        x_low: float = 0.0,
        x_high: float =  1,
        points: int = 1000,
        slope: float = 1.,
        intercept: float = 0.,
) -> LineCurve:
    x = np.linspace(x_low, x_high, points)
    y: npt.NDArray[np.float64] = np.array([intercept]) + slope*x
    return LineCurve(x=x, y=y)
