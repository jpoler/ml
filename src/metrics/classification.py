import numpy as np
import numpy.typing as npt
from operator import itemgetter
from typing import Tuple

def confusion_matrix(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return y_pred.T @ y_true

def precision(confusion_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # for brevity
    cm = confusion_matrix
    out = np.zeros(cm.shape[0])
    totals = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if totals[i] > 0:
            out[i] = cm[i, i] / totals[i]
    return out

def recall(confusion_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # for brevity
    cm = confusion_matrix
    out = np.zeros(cm.shape[0])
    totals = cm.sum(axis=0)
    for i in range(cm.shape[0]):
        if totals[i] > 0:
            out[i] = cm[i, i] / totals[i]
    return out

# I'm having a hard time seeing how precision_micro_average and
# recall_micro_average are different (from the follow link)
# https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/.
# In both cases TP_i comes from the diagonal and the denominator is the sum
# over the entire matrix.
def precision_micro_average(confusion_matrix: npt.NDArray[np.float64]) -> float:
    cm = confusion_matrix
    diag: npt.NDArray[np.float64] = np.diag(cm).sum() # type: ignore
    p: float = diag.sum() / cm.sum()
    return p

def recall_micro_average(confusion_matrix: npt.NDArray[np.float64]) -> float:
    cm = confusion_matrix
    diag: npt.NDArray[np.float64] = np.diag(cm).sum() # type: ignore
    p: float = diag.sum() / cm.sum()
    return p

def f1_micro(confusion_matrix: npt.NDArray[np.float64]) -> float:
    cm = confusion_matrix
    p = precision_micro_average(cm)
    r = recall_micro_average(cm)
    return 2 * (p*r) / (p+r)

def f1_macro(confusion_matrix: npt.NDArray[np.float64]) -> float:
    cm = confusion_matrix
    normalizer = float(cm.shape[0])
    precisions = precision(cm)
    recalls = recall(cm)
    p: float = precisions.sum() / normalizer
    r: float = recalls.sum() / normalizer
    return 2 * (p*r) / (p+r)


def multiclass_roc_auc(y_true: npt.NDArray[np.float64], ps: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64]]:
    """Multiclass ROC AUC originally described by Hand and Till.

    Based on this description: https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/
    """
    k = len(y_true[0, :])
    conditional_ps = np.zeros((k, k))
    for i in range(k):
        i_ps = ps[y_true[:, i] == 1.][:, i]
        ni = len(i_ps)
        for j in range(k):
            if i == j:
                continue
            j_ps = ps[y_true[:, j] == 1.][:, i]
            nj = len(j_ps)
            ranks = sorted(
                [(p, i) for p in i_ps] + [(p, j) for p in j_ps],
                key=itemgetter(0),
            )
            rank_totals = sum(r+1 for r, (_, k) in enumerate(ranks) if k == i)
            p_i_given_j = (rank_totals - ni*(ni+1)/2) / (ni*nj)
            conditional_ps[i,j] = p_i_given_j
    conditional_ps += conditional_ps.T
    conditional_ps *= 1/2.
    mask = np.zeros_like(conditional_ps, dtype=bool)
    idx = np.tril_indices_from(mask) # type: ignore
    mask[idx] = True
    conditional_ps[mask] = 0.
    total_auc = (2. / (k * (k - 1))) * np.sum(conditional_ps)
    return (total_auc, conditional_ps)
