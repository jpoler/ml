from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional

class LeastSquaresRegression(ABC):
    def __init__(self, regularization_coefficient: Optional[float] = None):
        self.regularization_coefficient = regularization_coefficient

    @abstractmethod
    def phi(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        pass

    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.float32]):
        phi = self.phi(X)
        if self.regularization_coefficient:
            tmp = np.linalg.inv(self.regularization_coefficient * np.eye(phi.T.shape[0]) @ phi.T @ phi)
        else:
            tmp = np.linalg.inv(phi.T @ phi)
        self.W = tmp @ phi.T @ y
        print(f"phi.shape {phi.shape}, self.W.shape {self.W.shape}")

    def predict(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        phi = self.phi(x)
        print(f"W shape {self.W.shape}, phi shape {phi.shape}")
        return self.W @ phi.T

class PolynomialBasisLeastSquaresRegression(LeastSquaresRegression):
    def __init__(self, m_degrees=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.M = m_degrees

    def phi(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.stack(list(np.array(list(np.power(xi, m) for m in range(self.M))) for xi in x))

def test():
    slope = 2
    origin = 1
    X = np.random.uniform(low=0.0, high=2*np.pi, size=100)
    noise_mean = 0
    noise_stddev = 0.2
    noise = np.random.normal(loc=noise_mean, scale=noise_stddev, size=100)
    y = 2*X + origin + noise

    r = PolynomialBasisLeastSquaresRegression()
    r.fit(X, y)

    X_sample = np.random.uniform(low=0.0, high=2*np.pi, size=100)
    y_pred = r.predict(X_sample)
    print(y_pred)

if __name__ == '__main__':
    test()

