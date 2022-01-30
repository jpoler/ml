from pytest import fixture
import numpy as np
from typing import List, Tuple

from data.sin import sin_data as sindata, SinData
from grid.hyperparameters import full_data_slices, ParameterSpace
from linear_regression import PolynomialBasisLeastSquaresRegression

@fixture
def sin_data() -> SinData:
    return sindata(n_train=1000, n_test=200)

@fixture
def sin_data_and_parameter_spaces(sin_data: SinData) -> Tuple[SinData, List[ParameterSpace]]:
    parameter_spaces = [
        ParameterSpace(
            model=PolynomialBasisLeastSquaresRegression,
            keyword="regularization_coefficient",
            base_parameters={"m_degrees": 10},
            space=np.linspace(0.0, 100.0, 10),
            base_data=sin_data,
            data_slices=full_data_slices(len(sin_data.x_train)),
        ),
        ParameterSpace(
            model=PolynomialBasisLeastSquaresRegression,
            keyword="m_degrees",
            base_parameters={},
            space=[i for i in range(10)],
            base_data=sin_data,
            data_slices=full_data_slices(len(sin_data.x_train)),
        ),
    ]
    return (sin_data, parameter_spaces)
