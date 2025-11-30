
from abc import ABC, abstractmethod
from dsbo.parameters import CompositeDesignParameter


class ConstraintFunctionBase(ABC):
    def __init__(self, binary: bool = False) -> None:
        super().__init__()
        self.binary_flag = binary

    def __call__(self, design: CompositeDesignParameter) -> tuple[float]:
        c = self.evaluate(design)
        return self.binary(c) if self.binary_flag else c

    @abstractmethod
    def evaluate(self, design: CompositeDesignParameter) -> tuple[float]:
        pass

    def binary(self, c: tuple[float]) -> tuple[float]:
        return tuple(min(1.0, max(0.0, v)) for v in c)
    

class MultiConstraint(ConstraintFunctionBase):
    def __init__(self, *constraints: ConstraintFunctionBase):
        assert not any(isinstance(c, MultiConstraint) for c in constraints), "Nested MultiConstraint is not supported"
        super().__init__(binary=False) #Just propopagate the output of the individual constraints
        self.constraints = constraints

    def evaluate(self, design: CompositeDesignParameter) -> tuple[float]:
        return tuple(v for c in self.constraints for v in c(design))