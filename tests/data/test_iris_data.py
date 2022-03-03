import numpy as np

from data.iris_data import iris_data

def test_iris_data() -> None:
    n_train_proportion = 0.5
    data = iris_data(n_train_proportion=n_train_proportion)
    assert len(data.x_train) == len(data.y_train)
    assert len(data.x_train) == len(data.x_test)
    assert len(data.x_train) == len(data.y_test)
    assert np.allclose(data.y_train.sum(axis=1), 1.)
    assert np.allclose(data.y_test.sum(axis=1), 1.)
