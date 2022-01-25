from abc import abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class FixedBasisFunctionMixin(Generic[T]):
    @abstractmethod
    def phi(self, x: T) -> T:
        pass
