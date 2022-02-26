from dataclasses import dataclass
from math import floor
import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal # type: ignore
from typing import Optional, Sequence

from data.data import Data

@dataclass
class GaussianClassData(Data):
    pass

def gaussian_class_data(
        means: Sequence[npt.NDArray[np.float64]],
        covariances: Sequence[npt.NDArray[np.float64]],
        proportions: Optional[Sequence[float]] = None,
        n_train: int = 30,
        n_test: int = 30,
) -> GaussianClassData:
    """Given gaussian means and covariances, generates gaussian class data.

    Returns a dataset sampled from the given distributions according to the specified proportions, labeled with a 1-of-K label."""
    if not proportions:
        u = 1. / float(len(means))
        proportions = [u for _ in range(len(means))]
    if len(means) != len(covariances) or len(means) != len(proportions):
        raise ValueError(f"shape mismatch: len(means): {len(means)}, len(covariances): {len(covariances)}, "
                         f"len(proportions): {len(proportions)}")

    mean_shapes = [m.shape for m in means]
    if not all(s == mean_shapes[0] for s in mean_shapes):
        raise ValueError("mean shape mismatch: {mean_shapes}")

    covariance_shapes = [c.shape for c in covariances]
    if not all(c == covariance_shapes[0] for c in covariance_shapes):
        raise ValueError(f"covariance shape mismatch: {covariance_shapes}")

    x = np.zeros((n_train + n_test, mean_shapes[0][0]))
    y = np.zeros((n_train + n_test, len(means)))

    total = 0
    for i, (mean, covariance, proportion) in enumerate(zip(means, covariances, proportions)):
        n = floor((n_train + n_test) * proportion)
        sample: npt.NDArray[np.float64] = multivariate_normal.rvs(mean=mean, cov=covariance, size=n)
        x[total:total+n] = sample
        one_hot = np.zeros((n, len(means)))
        one_hot[np.arange(n), i] = 1.
        y[total:total+n] = one_hot
        total += n

    p = np.random.default_rng().permutation(len(x))
    x_perm = x[p, :]
    y_perm = y[p, :]
    return GaussianClassData(x_train=x_perm[:n_train], x_test=x_perm[n_train:], y_train=y_perm[:n_train], y_test=y_perm[n_train:])
