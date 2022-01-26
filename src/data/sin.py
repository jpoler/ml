from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass
class SinData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]

@dataclass
class SinCurve:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

def sin_data(
        n_train: int = 80,
        n_test: int = 20,
        x_low: float = 0.,
        x_high: float = 2*np.pi,
        noise_stddev: float = 1.,
) -> SinData:
    rng = np.random.default_rng()
    n_total = n_train + n_test
    x = rng.uniform(low=x_low, high=x_high, size=n_total)
    noise = rng.normal(loc=0., scale=noise_stddev, size=n_total)
    y = np.sin(x) + noise
    return SinData(x_train=x[:n_train], y_train=y[:n_train], x_test=x[n_train+1:], y_test=y[n_train+1:])

def sin_curve(x_low: float = 0.0, x_high: float =  2*np.pi, points: int = 1000) -> SinCurve:
    x = np.linspace(x_low, x_high, points)
    y = np.sin(x)
    return SinCurve(x=x, y=y)
