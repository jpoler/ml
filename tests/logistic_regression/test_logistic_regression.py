import numpy as np
import pytest

from data.gaussian_classes import gaussian_class_data
from logistic_regression import PolynomialBasisLogisticRegression
from metrics.classification import confusion_matrix, precision, recall, multiclass_roc_auc

# @pytest.mark.focus
def test_logistic_regression() -> None:
    means = [
        np.array([0.0, 0.0]),
        np.array([3.0, 3.0]),
        np.array([-3, 3]),
    ]
    covariances = [
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
    ]
    k = len(means)
    ps = np.zeros(k)
    rs = np.zeros(k)
    overall_aucs = 0.
    pairwise_aucs = np.zeros((k, k))
    epochs = 30
    for i in range(epochs):
        data = gaussian_class_data(means, covariances, n_train=30, n_test=30)
        model = PolynomialBasisLogisticRegression(m_degrees=2, k_classes=len(data.y_train[1,:]))
        model.fit(data.x_train, data.y_train)
        y_pred = model.predict(data.x_test)
        cm = confusion_matrix(data.y_test, y_pred)
        ps += precision(cm)
        rs += recall(cm)
        print(f"confusion matrix: {cm}")
        print(f"precision: {ps}")
        print(f"recall: {rs}")
        S = model.predictive_probability(data.x_test)
        overall_auc, pairwise_auc = multiclass_roc_auc(data.y_test, S)
        overall_aucs += overall_auc
        pairwise_aucs += pairwise_auc
    print(f"average p: {(ps / float(epochs))}")
    print(f"average r: {(rs / float(epochs))}")
    print(f"average pairwise_aucs: {(pairwise_aucs / float(epochs))}")
    print(f"average overall_aucs: {(overall_aucs / float(epochs))}")
    assert np.all((ps / float(epochs)) > .5)
    assert np.all((rs / float(epochs)) > .5)
    for i in range(k):
        for j in range(i+1, k):
            assert (pairwise_aucs[i][j] / float(epochs)) > .8
    assert np.all((overall_aucs / float(epochs)) > .8)
