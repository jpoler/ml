from dataclasses import dataclass
from functools import partialmethod
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Sequence, Type, TypeVar

# bound is model?
C = TypeVar("C")

@dataclass
class PlotData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]


def class_partially_bound_parameters(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls

def plot_class_vs_parameter(plt, classes, param_name, params, plot_data) -> None:
    fig, axs = plt.subplots(len(params), len(classes), figsize=(8*len(classes), 8*len(params)))
    for c, cls in enumerate(classes):
        for r, p in enumerate(params):
            inst = cls(**{param_name:p})
            inst.fit(plot_data.x_train, plot_data.y_train)
            predictions = inst.predict(plot_data.x_test)
            axs[r, c].plot(plot_data.x_train, plot_data.y_train, "ro", plot_data.x_test, plot_data.y_test, "bo", plot_data.x_test, predictions, "gx")
            axs[r, c].set_title(f"class {cls.__name__} param: {p}")


    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
