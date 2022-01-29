from dataclasses import dataclass
# from functools import partialmethod
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Generator, Generic, Iterable, Optional, Type, TypeVar
from numbers import Number

from data.data import Data
from model import Model, T

# bound is model?
C = TypeVar("C")
# M = TypeVar("M", bound=Model[npt.NDArray[np.float64]])

@dataclass
class PlotData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]

@dataclass
class GridCell(Generic[T]):
    model: Type[Model[T]]
    data: Data[T]
    parameters: Dict[str, Any]
    predictions: Optional[T] = None

@dataclass
class ParameterSpace:
    keyword: str
    space: Iterable[Number]


# def class_partially_bound_parameters(cls, *args, **kwargs):
#     class NewCls(cls):
#         __init__ = partialmethod(cls.__init__, *args, **kwargs)
#     return NewCls

# def plot_class_vs_parameter(plt, classes, param_name, params, plot_data) -> None:
#     fig, axs = plt.subplots(len(params), len(classes), figsize=(8*len(classes), 8*len(params)))
#     for c, cls in enumerate(classes):
#         for r, p in enumerate(params):
#             inst = cls(**{param_name:p})
#             inst.fit(plot_data.x_train, plot_data.y_train)
#             predictions = inst.predict(plot_data.x_test)
#             axs[r, c].plot(plot_data.x_train, plot_data.y_train, "ro", plot_data.x_test, plot_data.y_test, "bo", plot_data.x_test, predictions, "gx")
#             axs[r, c].set_title(f"class {cls.__name__} param: {p}")


#     for ax in axs.flat:
#         ax.set(xlabel='x-label', ylabel='y-label')
#     # Hide x labels and tick labels for top plots and y ticks for right plots.
#     for ax in axs.flat:
#         ax.label_outer()

def parameter_generator(base_params: Dict[str, Any], param_space) -> Generator[Dict[str, Any], None, None]:
    for p in param_space.space:
        d = dict(base_params)
        d.update({param_space.keyword: p})
        yield d

def increasing_subslices(low: int, high: int, subsets: int, n: int) -> Generator[slice, None, None]:
    k = n // subsets
    for i in range(1, subsets+1):
        yield slice(0, i*k)

def full_data_slices(n: int) -> Generator[slice, None, None]:
    while True:
        yield slice(0, n)


def data_generator(base_data: Data[T], slices: Generator[slice, None, None]) -> Generator[Data[T], None, None]:
    # for brevity
    b = base_data
    for s in slices:
        yield Data(x_train=b.x_train[s], y_train=b.y_train[s], x_test=b.x_test[s], y_test=b.y_test[s])

def row_generator(model, data_generator, parameter_generator):
    for data, params in zip(data_generator, parameter_generator):
        yield GridCell(model=model, data=data, parameters=params)

def grid_generator(models, data_generators, parameter_generators):
    for model, data_generator, parameter_generator in zip(models, data_generators, parameter_generators):
        yield row_generator(model, data_generator, parameter_generator)

def grid_map(grid):
    for row in grid:
        yield [fit_and_predict(cell) for cell in row]

def full_data_grid(models, base_data, base_params, param_spaces):
    data_generators = [data_generator(base_data, full_data_slices(len(base_data.x_train))) for _ in range(len(models))]
    param_generators = [parameter_generator(base_params, param_space) for param_space in param_spaces ]
    return grid_map(grid_generator(models, data_generators, param_generators))

def plot_predictions(plt, grid):
    fig, axs = plt.subplots(len(grid[0]), len(grid), figsize=(8*len(grid), 8*len(grid[0])))
    for c, col in enumerate(grid):
        for r, cell in enumerate(col):
            axs[r, c].plot(cell.data.x_train, cell.data.y_train, "ro", cell.data.x_test, cell.data.y_test, "bo", cell.data.x_test, cell.predictions, "gx")
            axs[r, c].set_title(f"class {cell.model.__name__} params: {cell.parameters}")


    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()



def fit_and_predict(cell: GridCell[T]) -> GridCell[T]:
    inst = cell.model(**cell.parameters)
    inst.fit(cell.data.x_train, cell.data.y_train)
    cell.predictions = inst.predict(cell.data.x_test)
    return cell



def plot_data_grid():
    pass
