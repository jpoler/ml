from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Type

from data.data import Data
from metrics.regression import mean_squared_error
from metrics.classification import confusion_matrix, f1_macro
from model import M, GBM

Parameters = Dict[str, Any]

@dataclass
class PlotData:
    x_train: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    x_test: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]

@dataclass
class ParameterSpace(Generic[M]):
    model: Type[M]
    base_parameters: Parameters
    data: Iterable[Data]
    parameters: Iterable[Parameters]
    keyword: Optional[str] = None

    def __init__(self, model: Type[M], base_parameters: Parameters, space: Iterable[Any], base_data: Data, data_slices: Iterable[slice], keyword: Optional[str] = None) -> None:
        parameter_gen = parameter_generator(space, base_parameters, keyword)
        data_gen = data_generator(base_data, data_slices)
        self.model = model
        self.keyword = keyword
        self.base_parameters = base_parameters
        self.data = data_gen
        self.parameters = parameter_gen

@dataclass
class Cell(Generic[M]):
    parameter_space: ParameterSpace[M]
    data: Data
    parameters: Parameters


@dataclass
class EvaluatedCell(Cell[M]):
    model: M
    predictions: npt.NDArray[np.float64]

CellGridIterable = Iterable[Iterable[Cell[M]]]
EvaluatedCellGridIterable = Iterable[Iterable[EvaluatedCell[M]]]
EvaluatedCellGridList = List[List[EvaluatedCell[M]]]
EvaluatedCellGridSequence = Sequence[Sequence[EvaluatedCell[M]]]
MetricGridList = List[List[float]]
MetricGridSequence = Sequence[Sequence[float]]
MetricCallable = Callable[[EvaluatedCell], float]
GridMetricCallable = Callable[[EvaluatedCellGridIterable[M]], MetricGridList]

def parameter_generator(space: Iterable[Any], base_parameters: Parameters, keyword: Optional[str] = None) -> Iterable[Parameters]:
    for p in space:
        d = dict(base_parameters)
        if keyword:
            d.update({keyword: p})
        yield d

def single_slices(low: int, high: int) -> Iterable[slice]:
    for i in range(low, high):
        yield slice(low, low+1)

def increasing_subslices(low: int, high: int, subsets: int, include_empty: bool=False) -> Iterable[slice]:
    k = (high - low) // subsets
    start = 0 if include_empty else 1
    stop = start + subsets
    for i in range(start, stop):
        yield slice(low, low + i*k)


def full_data_slices(n: int) -> Iterable[slice]:
    while True:
        yield slice(0, n)

def data_generator(base_data: Data, slices: Iterable[slice]) -> Iterable[Data]:
    # for brevity
    b = base_data
    for s in slices:
        yield Data(x_train=b.x_train[s], y_train=b.y_train[s], x_test=b.x_test[s], y_test=b.y_test[s])

def row_generator(parameter_space: ParameterSpace[M], data_generator: Iterable[Data], parameter_generator: Iterable[Parameters]) -> Iterable[Cell[M]]:
    for data, params in zip(data_generator, parameter_generator):
        yield Cell(parameter_space=parameter_space, data=data, parameters=params)

def grid_generator(parameter_spaces: Iterable[ParameterSpace[M]]) -> CellGridIterable[M]:
    for parameter_space in parameter_spaces:
        yield row_generator(parameter_space, parameter_space.data, parameter_space.parameters)

def grid_map(grid: CellGridIterable[M]) -> EvaluatedCellGridIterable[M]:
    for row in grid:
        yield [fit_and_predict(cell) for cell in row]

def expand_grid(parameter_spaces: Iterable[ParameterSpace[M]]) -> EvaluatedCellGridList[M]:
    gen = grid_generator(parameter_spaces)
    return list(list(row) for row in grid_map(gen))

def compute_metric(grid: EvaluatedCellGridIterable[M], metric: MetricCallable) -> MetricGridList:
    return [[metric(cell) for cell in row] for row in grid]

def mean_squared_error_metric_cell(cell: EvaluatedCell[M]) -> float:
    return mean_squared_error(cell.data.y_test, cell.predictions)

def mean_squared_error_metric(grid: EvaluatedCellGridIterable[M]) -> MetricGridList:
    return compute_metric(grid, mean_squared_error_metric_cell)

def f1_macro_metric_cell(cell: EvaluatedCell[M]) -> float:
    cm = confusion_matrix(cell.data.y_test, cell.predictions)
    return f1_macro(cm)

def f1_macro_metric(grid: EvaluatedCellGridIterable[M]) -> MetricGridList:
    return compute_metric(grid, f1_macro_metric_cell)

def fit_and_predict(cell: Cell[M]) -> EvaluatedCell[M]:
    inst = cell.parameter_space.model(**cell.parameters)
    inst.fit(cell.data.x_train, cell.data.y_train)
    predictions = inst.predict(cell.data.x_test)
    model = inst
    return EvaluatedCell(parameter_space=cell.parameter_space, data=cell.data, parameters=cell.parameters, predictions=predictions, model=model)

# todo find type of plt
def plot_predictions(plt: Any, grid: EvaluatedCellGridSequence[M]) -> None:
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

def plot_classification_predictions(plt: Any, grid: EvaluatedCellGridSequence[M]) -> None:
    fig, axs = plt.subplots(len(grid[0]), len(grid), figsize=(8*len(grid), 8*len(grid[0])))
    for c, col in enumerate(grid):
        for r, cell in enumerate(col):
            minx = cell.data.x_train[:, 0].min() - 1
            maxx = cell.data.x_train[:, 0].max() + 1
            miny = cell.data.x_train[:, 1].min() - 1
            maxy = cell.data.x_train[:, 1].max() + 1
            xrange = np.arange(minx, maxx, 0.1)
            yrange = np.arange(miny, maxy, 0.1)
            xx, yy = np.meshgrid(xrange, yrange)  # type: ignore
            xx_flat = xx.flatten()
            yy_flat = yy.flatten()
            xx_flat = xx_flat.reshape((len(xx_flat), 1))
            yy_flat = yy_flat.reshape((len(yy_flat), 1))
            np_grid = np.hstack((xx_flat, yy_flat))
            predictions = np.argmax(cell.model.predict(np_grid), axis=1)
            zz = predictions.reshape(xx.shape)
            axs[r,c].contourf(xx, yy, zz, cmap="Paired")
            for k in range(3):
                row_ix = np.where(cell.data.y_train[:, k] == 1.0)
                axs[r,c].scatter(cell.data.x_train[row_ix, 0], cell.data.x_train[row_ix, 1], cmap="Paired")

            param_string = "\n".join(sorted(f"{k}: {v}" for k, v in cell.parameters.items()))
            axs[r, c].set_title(f"{cell.parameter_space.model.__name__}\n{param_string}")


    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='y')
    for ax in axs.flat:
        ax.label_outer()

def max_eigenvalue(cell: EvaluatedCell[GBM]) -> np.float64:
    covariance = cell.model.posterior_covariance()
    # a real, symmetric matrix will have real eigenvalues
    complex_eigenvalues, _ = np.linalg.eig(covariance) # type: ignore
    real_eigenvalues: npt.NDArray[np.float64] = np.real(complex_eigenvalues) # type: ignore
    return max(real_eigenvalues)

def get_mean(cell: EvaluatedCell[GBM]) -> npt.NDArray[np.float64]:
    return cell.model.posterior_mean()

def plot_bayesian_probabilities(plt: Any, grid: EvaluatedCellGridSequence[GBM]) -> None:
    if len(grid) < 1:
        raise ValueError("expected at least 1 column")
    if len(grid[0]) < 2:
        raise ValueError("expected at least 2 rows")
    fig, axs = plt.subplots(len(grid[0]), 3*len(grid), figsize=(24*len(grid), 8*len(grid[0])))
    for c, col in enumerate(grid):
        data_per_iteration = len(col[1].data.x_train) - len(col[0].data.x_train)
        radius = max(map(max_eigenvalue, col))*3
        average_mean = np.add.reduce(list(map(get_mean, col))) / float(len(col))
        x = np.linspace(average_mean[0] - radius, average_mean[0] + radius, 50)
        y = np.linspace(average_mean[1] - radius, average_mean[1] + radius, 50)
        X, Y = np.meshgrid(x, y) # type: ignore
        for r, cell in enumerate(col):
            if len(cell.data.x_train) == 0:
                continue
            model = cell.model
            Z = np.empty(X.shape)
            for ix, iy in np.ndindex(Z.shape):
                Z[ix, iy] = model.likelihood_probability(
                    cell.data.x_train[-data_per_iteration:],
                    cell.data.y_train[-data_per_iteration:],
                    np.array([X[ix, iy], Y[ix, iy]]),
                )
            axs[r,c].contourf(X, Y, Z)
            axs[r,c].set_aspect('equal')
            axs[r,c].set_title("parameter space\nlikelihood probability of added data")
            axs[r,c].set(xlabel="w0", ylabel="w1")
        for r, cell in enumerate(col):
            model = cell.model
            Z = np.empty(X.shape)
            for ix, iy in np.ndindex(Z.shape):
                Z[ix, iy] = model.posterior_probability(np.array([X[ix, iy], Y[ix, iy]]))
            axs[r,c+1].contourf(X, Y, Z)
            axs[r,c+1].set_aspect('equal')
            axs[r,c+1].set_title("parameter space\nposterior probability")
            axs[r,c+1].set(xlabel="w0", ylabel="w1")
        for r, cell in enumerate(col):
            for i in range(6):
                model = cell.model
                w = model.sample_posterior_parameters()
                x = np.linspace(0, 1, 50)
                b = np.ones(50)
                X = np.vstack((b, x))
                y = X.T @ w
                axs[r,c+2].plot(x, y, "r-", cell.data.x_train, cell.data.y_train, "bo")
                axs[r,c+2].set_title("data space")
                axs[r,c+2].set(xlabel="x", ylabel="y")


# todo find type of plt
def plot_metrics(plt: Any, grid: EvaluatedCellGridSequence[M], metric: GridMetricCallable[M]) -> None:
    metrics = metric(grid)
    fig, axs = plt.subplots(1, len(metrics), figsize=(8*len(grid), 8))
    for c, m in enumerate(metrics):
        parameter_space = grid[c][0].parameter_space
        if parameter_space.keyword:
            p = [cell.parameters[parameter_space.keyword] for cell in grid[c]]
        else:
            p = [len(cell.data.x_train) for cell in grid[c]]
        axs[c].plot(p, m)
        axs[c].set_title(f"{parameter_space.model.__name__}\n{parameter_space.keyword}")
        axs[c].set(xlabel=parameter_space.keyword, ylabel=metric.__name__)
    for ax in axs.flat:
        ax.label_outer()
