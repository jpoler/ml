from dataclasses import dataclass
# from functools import partialmethod
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type

from data.data import Data
from metrics.regression import mean_squared_error
from model import Model

Parameters = Dict[str, Any]

@dataclass
class PlotData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]

@dataclass
class ParameterSpace:
    model: Type[Model]
    keyword: str
    base_parameters: Parameters
    data: Iterable[Data]
    parameters: Iterable[Parameters]

    def __init__(self, model: Type[Model], keyword: str, base_parameters: Parameters, space: Iterable[Any], base_data: Data, data_slices: Iterable[slice]) -> None:
        parameter_gen = parameter_generator(space, keyword, base_parameters)
        data_gen = data_generator(base_data, data_slices)
        self.model = model
        self.keyword = keyword
        self.base_parameters = base_parameters
        self.data = data_gen
        self.parameters = parameter_gen

@dataclass
class Cell:
    parameter_space: ParameterSpace
    data: Data
    parameters: Parameters
    predictions: Optional[npt.NDArray[np.float64]] = None

CellGridIterable = Iterable[Iterable[Cell]]
CellGridList = List[List[Cell]]
CellGridSequence = Sequence[Sequence[Cell]]
MetricGridList = List[List[float]]
MetricGridSequence = Sequence[Sequence[float]]
MetricCallable = Callable[[Cell], float]
GridMetricCallable = Callable[[CellGridIterable], MetricGridList]

def parameter_generator(space: Iterable[Any], keyword: str, base_parameters: Parameters) -> Iterable[Parameters]:
    for p in space:
        d = dict(base_parameters)
        d.update({keyword: p})
        yield d

def increasing_subslices(low: int, high: int, subsets: int) -> Iterable[slice]:
    k = (high - low) // subsets
    for i in range(1, subsets+1):
        yield slice(low, low + i*k)

def full_data_slices(n: int) -> Iterable[slice]:
    while True:
        yield slice(0, n)

def data_generator(base_data: Data, slices: Iterable[slice]) -> Iterable[Data]:
    # for brevity
    b = base_data
    for s in slices:
        yield Data(x_train=b.x_train[s], y_train=b.y_train[s], x_test=b.x_test[s], y_test=b.y_test[s])

def row_generator(parameter_space: ParameterSpace, data_generator: Iterable[Data], parameter_generator: Iterable[Parameters]) -> Iterable[Cell]:
    for data, params in zip(data_generator, parameter_generator):
        yield Cell(parameter_space=parameter_space, data=data, parameters=params)

def grid_generator(parameter_spaces: Iterable[ParameterSpace]) -> CellGridIterable:
    for parameter_space in parameter_spaces:
        yield row_generator(parameter_space, parameter_space.data, parameter_space.parameters)

def grid_map(grid: CellGridIterable) -> CellGridIterable:
    for row in grid:
        yield [fit_and_predict(cell) for cell in row]

def expand_grid(parameter_spaces: Iterable[ParameterSpace]) -> CellGridList:
    gen = grid_generator(parameter_spaces)
    return list(list(row) for row in grid_map(gen))

def compute_metric(grid: CellGridIterable, metric: MetricCallable) -> MetricGridList:
    return [[metric(cell) for cell in row] for row in grid]

def mean_squared_error_metric_cell(cell: Cell) -> float:
    if cell.predictions is None:
        raise ValueError("expected predictions to be present")
    return mean_squared_error(cell.data.y_test, cell.predictions)

def mean_squared_error_metric(grid: CellGridIterable) -> MetricGridList:
    return compute_metric(grid, mean_squared_error_metric_cell)

def fit_and_predict(cell: Cell) -> Cell:
    inst = cell.parameter_space.model(**cell.parameters)
    inst.fit(cell.data.x_train, cell.data.y_train)
    cell.predictions = inst.predict(cell.data.x_test)
    return cell

# todo find type of plt
def plot_predictions(plt: Any, grid: CellGridSequence) -> None:
    # reveal_type(plt)
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

# todo find type of plt
def plot_metrics(plt: Any, grid: CellGridSequence, metric: GridMetricCallable) -> None:
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
