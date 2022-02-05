from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class Model(ABC):
    @abstractmethod
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    # TODO change to X to reflect that it could be a design matrix
    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        pass

class BayesianModel(Model):
    @abstractmethod
    def posterior_probability(self, w: npt.NDArray[np.float64]) -> float:
        pass

    # # TODO make this a typevar with generics
    # @abstractmethod
    # def posterior_parameters(self) -> Any:
    #     pass

    @abstractmethod
    def likelihood_probability(self, X: npt.NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def predictive_probability(self, t: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
        pass
