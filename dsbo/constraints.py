
from abc import ABC, abstractmethod
from typing import Sequence
from dsbo.parameters import CompositeDesignParameter

class ConstraintBase(ABC):
    def __init__(self, binary: bool = False):
        super().__init__()
        self.binary_flag = binary

    def binary(self, c: Sequence[float]) -> Sequence[float]:
        return tuple(min(1.0, max(0.0, v)) for v in c)
    
    def __repr__(self) -> str:
        return self.__class__.__name__


class GeometryConstraintBase(ConstraintBase):
    def __init__(self, binary: bool = False) -> None:
        super().__init__(binary)

    def __call__(self, design: dict) -> Sequence[float]:
        c = self.evaluate(design)
        return self.binary(c) if self.binary_flag else c

    @abstractmethod
    def evaluate(self, design: dict) -> Sequence[float]:
        pass


class MultiConstraint(GeometryConstraintBase):
    def __init__(self, *constraints: GeometryConstraintBase):
        assert not any(isinstance(c, MultiConstraint) for c in constraints), "Nested MultiConstraint is not supported"
        super().__init__(binary=False) #Just propopagate the output of the individual constraints
        self.constraints = constraints
        self.lookup = {str(c): c for c in constraints}

    def evaluate(self, design: dict) -> Sequence[float]:
        return tuple(v for c in self.constraints for v in c(design))
    
    def get_by_name(self, name: str) -> GeometryConstraintBase:
        return self.lookup[name]