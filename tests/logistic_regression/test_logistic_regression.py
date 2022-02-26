import numpy as np
import pytest

from data.gaussian_classes import gaussian_class_data
from logistic_regression import PolynomialBasisLogisticRegression

@pytest.mark.focus
def test_logistic_regression() -> None:
    means = [
        np.array([-1., 1.]),
        np.array([1., 1.]),
       ]
    covariances = [
        np.array([
            [1., 0.],
            [0., 1.],
        ]),
        np.array([
            [1., 0.],
            [0., 1.],
        ])
       ]
    data = gaussian_class_data(means, covariances, n_train=10, n_test=10)
    model = PolynomialBasisLogisticRegression(m_degrees=2, k_classes=len(data.y_train[1,:]))
    model.fit(data.x_train, data.y_train)
    assert False
