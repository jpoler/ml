from dataclasses import dataclass
from typing import Generic

from model import T

@dataclass
class Data(Generic[T]):
    x_train: T
    y_train: T
    x_test: T
    y_test: T
