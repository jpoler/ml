from abc import abstractmethod
from typing import TypeVar, Generic, Sequence
from numbers import Number

# E = TypeVar("E")
T = TypeVar("T", bound=Sequence[Number])

class Model(Generic[T]):
    @abstractmethod
    def fit(self, X: T, y: T) -> None:
        pass

    @abstractmethod
    def predict(self, x: T) -> T:
        pass
