from dataclasses import dataclass
# from functools import partialmethod
import numpy as np
import numpy.typing as npt
from typing import Any, Dict, Generator, Generic, Iterable, Optional, Type, TypeVar
from numbers import Number

from data.data import Data
from metrics.regression import mean_squared_error
from model import Model, T

# bound is model?
C = TypeVar("C")
# M = TypeVar("M", bound=Model[npt.NDArray[np.float64]])
Parameters = Dict[str, Any]

@dataclass
class PlotData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]

@dataclass
class ParameterSpace(Generic[T]):
    model: Type[Model[T]]
    keyword: str
    base_parameters: Parameters
    data: Generator[Data[T], None, None]
    parameters: Generator[Parameters, None, None]

    def __init__(self, model, keyword, base_parameters, space, data):
        parameter_gen = parameter_generator(space, keyword, base_parameters)
        self.model = model
        self.keyword = keyword
        self.base_parameters = base_parameters
        self.data = data
        self.parameters = parameter_gen

@dataclass
class GridCell(Generic[T]):
    parameter_space: ParameterSpace[T]
    data: Data[T]
    parameters: Parameters
    predictions: Optional[T] = None




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

def parameter_generator(space, keyword, base_parameters) -> Generator[Parameters, None, None]:
    for p in space:
        d = dict(base_parameters)
        d.update({keyword: p})
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

def row_generator(parameter_space, data_generator, parameter_generator):
    for data, params in zip(data_generator, parameter_generator):
        yield GridCell(parameter_space=parameter_space, data=data, parameters=params)

def grid_generator(parameter_spaces):
    for parameter_space in parameter_spaces:
        yield row_generator(parameter_space, parameter_space.data, parameter_space.parameters)

def grid_map(grid):
    for row in grid:
        yield [fit_and_predict(cell) for cell in row]

def full_data_parameter_space(model, keyword, base_parameters, space, base_data):
    data_gen = data_generator(base_data, full_data_slices(len(base_data.x_train)))
    return ParameterSpace(model=model, keyword=keyword, base_parameters=base_parameters, space=space, data=data_gen)

def full_data_grid(parameter_spaces):
    gen = grid_generator(parameter_spaces)
    return list(grid_map(gen))

def plot_predictions(plt, grid):
    fig, axs = plt.subplots(len(grid[0]), len(grid), figsize=(8*len(grid), 8*len(grid[0])))
    for c, col in enumerate(grid):
        for r, cell in enumerate(col):
            axs[r, c].plot(cell.data.x_train, cell.data.y_train, "ro", cell.data.x_test, cell.data.y_test, "bo", cell.data.x_test, cell.predictions, "gx")
            param_string = "\n".join(sorted(f"{k}: {v:.{2}f}" for k, v in cell.parameters.items()))
            axs[r, c].set_title(f"{cell.parameter_space.model.__name__}\n{param_string}")


    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')
    for ax in axs.flat:
        ax.label_outer()

def compute_metric(grid, metric):
    return [[metric(cell) for cell in row] for row in grid]

def plot_metrics(plt, grid, metric):
    metrics = metric(grid)
    fig, axs = plt.subplots(1, len(metrics), figsize=(8*len(grid), 8))
    for c, m in enumerate(metrics):
        parameter_space = grid[c][0].parameter_space
        p = [cell.parameters[parameter_space.keyword] for cell in grid[c]]
        axs[c].plot(p, m)
        axs[c].set_title(f"{parameter_space.model.__name__}\n{parameter_space.keyword}")
        axs[c].set(xlabel=parameter_space.keyword, ylabel=metric.__name__)
    for ax in axs.flat:
        ax.label_outer()

def mean_squared_error_metric_cell(cell):
    return mean_squared_error(cell.data.y_test, cell.predictions)

def mean_squared_error_metric(grid):
    return compute_metric(grid, mean_squared_error_metric_cell)

def fit_and_predict(cell: GridCell[T]) -> GridCell[T]:
    inst = cell.parameter_space.model(**cell.parameters)
    inst.fit(cell.data.x_train, cell.data.y_train)
    cell.predictions = inst.predict(cell.data.x_test)
    return cell



def plot_data_grid():
    pass
