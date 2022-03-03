import numpy as np
import numpy.typing as npt
import pytest
from typing import List, Tuple

from metrics.classification import confusion_matrix, f1_macro, f1_micro, precision, recall, multiclass_roc_auc

def input_data() -> List[List[npt.NDArray[np.float64]]]:
    return [
        [
            np.eye(3),
            np.fromiter((1. if i == j else 0.
                         for i in range(3)
                         for j in range(3)
                         for _ in range(3)), float).reshape((9,3), order="F"), # type: ignore
            np.fromiter((1. if i == j else 0.
                         for _ in range(3)
                         for i in range(3)
                         for j in range(3)), float).reshape((9,3)), # type: ignore
            np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        ],
        [
            # all correct predictions
            np.eye(3),
            # 1 correct, k-1 incorrect per class
            np.fromiter((1. if i == j else 0.
                         for _ in range(3)
                         for i in range(3)
                         for j in range(3)), float).reshape((9,3)), # type: ignore
            # predict the same class every time
            np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
            # 1 correct, k-1 incorrect predictions for each class when the true class is always the same
            np.fromiter((1. if i == j else 0.
                         for _ in range(3)
                         for i in range(3)
                         for j in range(3)), float).reshape((9,3)), # type: ignore
        ]

    ]

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [
    np.eye(3),
    np.ones((3,3)),
    np.vstack([3*np.ones((1, 3)), np.zeros((2, 3))]),
    np.hstack([3*np.ones((3, 1)), np.zeros((3, 2))]),
])))
def test_confusion_matrix(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, expected)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [
    np.ones(3),
    np.full(3, 1/3.),
    np.array([1/3., 0., 0.]),
    np.array([1., 0., 0.]),
])))
def test_precision(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    p = precision(cm)
    assert np.array_equal(p, expected)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [
    np.ones(3),
    np.full(3, 1/3.),
    np.array([1., 0., 0.]),
    np.array([1/3., 0., 0.]),
])))
def test_recall(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    p = recall(cm)
    assert np.array_equal(p, expected)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [1., 1./3., 1./3., 1./3.])))
def test_f1_micro(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_micro(cm)
    assert np.isclose(f1, expected, rtol=0)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [1., 1./3., 1./6., 1./6.])))
def test_f1_macro(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_macro(cm)
    assert np.isclose(f1, expected, rtol=0)

@pytest.mark.skip("ties need to be resolved")
@pytest.mark.parametrize("inputs", [
    (
        np.eye(3),
        np.eye(3),
        1.,
        np.array([
            [0., 1., 1.],
            [0., 0., 1.],
            [0., 0., 0.],
        ])
    ),
    # todo need to solve the tie edge case
    # (
    #     np.eye(3),
    #     (1./3.)*np.ones((3,3)),
    #     1.,
    #     np.array([
    #         [0., 1., 1.],
    #         [0., 0., 1.],
    #         [0., 0., 0.],
    #     ])
    # ),
])
def test_multiclass_roc_auc(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64]]) -> None:
    y_true, S, expected_overall_auc, expected_conditional_ps = inputs
    total_auc, conditional_ps = multiclass_roc_auc(y_true, S)
    assert np.isclose(total_auc, 0., rtol=0.)
    assert np.allclose(conditional_ps, expected_conditional_ps, rtol=0.)
