from abc import abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class Model(Generic[T]):
    @abstractmethod
    def fit(self, X: T, y: T) -> None:
        pass

    def predict(self, x: T) -> T:
        pass
