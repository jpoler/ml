from functools import partial
import numpy as np
import pytest
from typing import Callable

from data.data import Data
from data.iris_data import iris_data
from data.gaussian_classes import gaussian_class_data
from logistic_regression import PolynomialBasisLogisticRegression
from metrics.classification import confusion_matrix, f1_macro, f1_micro, precision, recall, multiclass_roc_auc
from model import SC

gaussian_data = partial(gaussian_class_data,
                        means=[
                            np.array([0.0, 0.0]),
                            np.array([3.0, 3.0]),
                            np.array([-3, 3]),
                        ],
                        covariances=[
                            np.array(
                                [
                                    [1.0, 0.0],
                                    [0.0, 1.0],
                                ]
                            ),
                            np.array(
                                [
                                    [2.0, 0.0],
                                    [0.0, 2.0],
                                ]
                            ),
                            np.array(
                                [
                                    [1.0, 0.0],
                                    [0.0, 1.0],
                                ]
                            ),
                        ],
                        n_train=30,
                        n_test=30)


@pytest.mark.focus
@pytest.mark.parametrize("model_init", [
    partial(PolynomialBasisLogisticRegression, m_degrees=2),
])
@pytest.mark.parametrize("data_init", [
    gaussian_data,
    iris_data,
])
def test_classification_models(model_init: Callable[[], SC], data_init: Callable[[], Data]) -> None:
    # initialize data once to understand its shape
    data = data_init()
    k = len(data.y_test[0, :])
    ps = np.zeros(k)
    rs = np.zeros(k)
    overall_aucs = 0.
    f1_micros = 0.
    f1_macros = 0.
    pairwise_aucs = np.zeros((k, k))
    epochs = 10
    for i in range(epochs):
        data = data_init()
        model = model_init()
        model.fit(data.x_train, data.y_train)
        y_pred = model.predict(data.x_test)
        cm = confusion_matrix(data.y_test, y_pred)
        ps += precision(cm)
        rs += recall(cm)
        f1_micros += f1_micro(cm)
        f1_macros += f1_macro(cm)
        print(f"confusion matrix: {cm}")
        print(f"precision: {ps}")
        print(f"recall: {rs}")
        S = model.soft_predict(data.x_test)
        overall_auc, pairwise_auc = multiclass_roc_auc(data.y_test, S)
        overall_aucs += overall_auc
        pairwise_aucs += pairwise_auc
    print(f"average p: {(ps / float(epochs))}")
    print(f"average r: {(rs / float(epochs))}")
    print(f"average f1_micro: {(f1_micros / float(epochs))}")
    print(f"average f1_macro: {(f1_macros / float(epochs))}")
    print(f"average pairwise_aucs: {(pairwise_aucs / float(epochs))}")
    print(f"average overall_aucs: {(overall_aucs / float(epochs))}")
    assert np.all((ps / float(epochs)) > .5)
    assert np.all((rs / float(epochs)) > .5)
    assert np.all((f1_micros / float(epochs)) > .7)
    assert np.all((f1_macros / float(epochs)) > .7)
    for i in range(k):
        for j in range(i+1, k):
            assert (pairwise_aucs[i][j] / float(epochs)) > .7
    assert np.all((overall_aucs / float(epochs)) > .7)
