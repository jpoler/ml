import numpy as np

from data.gaussian_classes import gaussian_class_data

def test_gaussian_class_data() -> None:
    means = [
        np.array([0., 0.]),
        np.array([5., 5.]),
    ]
    covariances = [
        np.array([
            [1., 0.],
            [0., 1.],
        ]),
        np.array([
            [2., 0.],
            [0., 2.],
        ])
    ]
    data = gaussian_class_data(means, covariances, n_train=20, n_test=10)
    assert len(data.x_train) == 20
    assert len(data.x_test) == 10

    assert np.isclose(data.y_train.sum(), 20.)
    assert np.isclose(data.y_test.sum(), 10.)
