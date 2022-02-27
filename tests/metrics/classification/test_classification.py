import numpy as np
import numpy.typing as npt
import pytest
from typing import List, Tuple

from metrics.classification import confusion_matrix, f1_macro, f1_micro, precision, recall, multiclass_roc_auc

def input_data() -> List[List[npt.NDArray[np.float64]]]:
    return [
        [
            np.eye(3),
            np.fromiter((1. if i == j else 0. for i in range(3) for j in range(3) for _ in range(3)), float).reshape((9,3), order="F"), # type: ignore
            np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
            np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        ],
        [
            # all correct predictions
            np.eye(3),
            # 1 correct, k-1 incorrect per class
            np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
            # predict the same class every time
            np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
            # 1 correct, k-1 incorrect predictions for each class when the true class is always the same
            np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        ]

    ]

@pytest.mark.parametrize("inputs", [
    # all correct predictions
    (
        np.eye(3), np.eye(3), np.eye(3),
    ),
    # 1 correct, k-1 incorrect per class
    (
        np.fromiter((1. if i == j else 0. for i in range(3) for j in range(3) for _ in range(3)), float).reshape((9,3), order="F"), # type: ignore
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.ones((3,3)),
    ),
    # predict the same class every time
    (
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.vstack([3*np.ones((1, 3)), np.zeros((2, 3))]),
    ),
    # 1 correct, k-1 incorrect predictions for each class when the true class is always the same
    (
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.hstack([3*np.ones((3, 1)), np.zeros((3, 2))]),
    ),
])
def test_confusion_matrix(inputs):
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    assert np.array_equal(cm, expected)

@pytest.mark.parametrize("inputs", [
    # all correct predictions
    (
        np.eye(3), np.eye(3), np.ones(3),
    ),
    # 1 correct, k-1 incorrect per class
    (
        np.fromiter((1. if i == j else 0. for i in range(3) for j in range(3) for _ in range(3)), float).reshape((9,3), order="F"), # type: ignore
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.full(3, 1/3.),
    ),
    # predict the same class every time
    (
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.array([1/3., 0., 0.]),
    ),
    # 1 correct, k-1 incorrect predictions for each class when the true class is always the same
    (
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.array([1., 0., 0.]),
    ),
])
def test_precision(inputs):
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    p = precision(cm)
    assert np.array_equal(p, expected)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [1., 1./3., 1./3., 1./3.])))
@pytest.mark.focus
def test_f1_micro(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_micro(cm)
    assert np.isclose(f1, expected, rtol=0)

@pytest.mark.parametrize("inputs", list(zip(*input_data(), [1., 1./3., 1./6., 1./6.])))
@pytest.mark.focus
def test_f1_macro(inputs: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    print(f"cm {cm}")
    f1 = f1_macro(cm)
    assert np.isclose(f1, expected, rtol=0)


@pytest.mark.parametrize("inputs", [
    # all correct predictions
    (
        np.eye(3), np.eye(3), np.ones(3),
    ),
    # 1 correct, k-1 incorrect per class
    (
        np.fromiter((1. if i == j else 0. for i in range(3) for j in range(3) for _ in range(3)), float).reshape((9,3), order="F"), # type: ignore
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.full(3, 1/3.),
    ),
    # predict the same class every time
    (
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.array([1., 0., 0.]),
    ),
    # 1 correct, k-1 incorrect predictions for each class when the true class is always the same
    (
        np.hstack([np.ones((9, 1)), np.zeros((9, 2))]),
        np.fromiter((1. if i == j else 0. for _ in range(3) for i in range(3) for j in range(3)), float).reshape((9,3)), # type: ignore
        np.array([1/3., 0., 0.]),
    ),
])
def test_recall(inputs) -> None:
    y_true, y_pred, expected = inputs
    cm = confusion_matrix(y_true, y_pred)
    p = recall(cm)
    assert np.array_equal(p, expected)

def test_multiclass_roc_auc() -> None:
    S: npt.NDArray[np.float64] = np.array(
        [[7.50765762e-04, 9.99249234e-01, 5.27420285e-16],
         [6.68713739e-03, 3.45011983e-06, 9.93309412e-01],
         [9.48430145e-01, 5.15697958e-02, 5.95110516e-08],
         [9.76641617e-01, 2.33583832e-02, 2.89905025e-15],
         [1.53688306e-12, 3.36299852e-13, 1.00000000e+00],
         [4.33164417e-02, 9.56683558e-01, 2.56420689e-14],
         [1.36743730e-11, 4.55989201e-13, 1.00000000e+00],
         [1.43075241e-09, 1.04525093e-09, 9.99999998e-01],
         [5.50264098e-01, 4.49735897e-01, 5.32470571e-09],
         [1.90452101e-10, 1.08999642e-12, 1.00000000e+00],
         [9.85870716e-01, 1.41292168e-02, 6.68767405e-08],
         [6.59878001e-04, 9.99340122e-01, 3.43148539e-12],
         [9.72877725e-01, 2.71222748e-02, 1.02564685e-15],
         [4.01029262e-04, 9.99598971e-01, 7.50088848e-28],
         [9.65095368e-01, 3.49046322e-02, 4.24549710e-17],
         [3.23329208e-05, 9.99967667e-01, 6.54557021e-21],
         [1.14603594e-06, 9.52641747e-08, 9.99998759e-01],
         [9.91890934e-01, 8.10856605e-03, 5.00250313e-07],
         [4.55338597e-04, 9.99544661e-01, 4.14639995e-18],
         [8.13829878e-01, 1.85201884e-01, 9.68237738e-04],
         [3.72734472e-06, 1.95284924e-07, 9.99996077e-01],
         [8.20581250e-08, 2.10304873e-07, 9.99999708e-01],
         [1.40528789e-01, 8.59471211e-01, 1.61709569e-18],
         [9.62277404e-01, 3.77225957e-02, 6.99327774e-11],
         [2.04974885e-05, 9.99979503e-01, 1.49536017e-16],
         [9.42781025e-01, 5.72189747e-02, 1.24027595e-13],
         [1.82535556e-07, 3.72486548e-09, 9.99999814e-01],
         [1.82942365e-04, 9.99817058e-01, 3.48600359e-16],
         [9.51019815e-01, 4.89801849e-02, 2.19787899e-20],
         [6.81679864e-12, 5.81622627e-14, 1.00000000e+00]],
    )
    y: npt.NDArray[np.float64] = np.array(
        [[0., 1., 0.],
         [0., 0., 1.],
         [1., 0., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0., 1., 0.],
         [0., 0., 1.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0., 0., 1.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.],
         [0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.]],
    )

    total_auc, conditional_ps = multiclass_roc_auc(y, S)
    assert np.isclose(total_auc, 1., rtol=0.)
    k = len(y[0, :])
    for i in range(k):
        for j in range(i+1, k):
            assert np.isclose(conditional_ps[i][j], 1., rtol=0.)
