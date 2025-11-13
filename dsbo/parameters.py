from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple
import numpy as np

class DesignParameter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.current_value: Optional[Any] = None

    @abstractmethod
    def float_repr(self) -> str:
        pass

class CompositeDesignParameter(DesignParameter):
    def __init__(self, name: str, parameters: Sequence[DesignParameter]) -> None:
        super().__init__(name)
        self._parameters = parameters

    def to_dict(self, flatten: bool = False) -> dict:
        if flatten:
            result = {}
            for param in self._parameters:
                if (isinstance(param, CompositeDesignParameter) and not 
                    isinstance(param, VectorDesignParameter)):
                    result.update(param.to_dict(flatten=True))
                elif isinstance(param, VectorDesignParameter):
                    result[param.name] = param.to_numpy()
                else:
                    result[param.name] = param.current_value
            return result
        else:
            # Preserve hierarchy
            result = {}
            for param in self._parameters:
                if isinstance(param, CompositeDesignParameter):
                    result[param.name] = param.to_dict(flatten=False)
                else:
                    result[param.name] = param.current_value
            return result
        

    def __repr__(self) -> str:
        return f"CompositeDesignParameter(name={self.name}, parameters={self._parameters})"


class ContinuousDesignParameter(DesignParameter):
    def __init__(self, name, min_value: float, max_value: float, value: Optional[float] = 0) -> None:
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = (max_value - min_value)/2

    def __repr__(self) -> str:
        return f"ContinuousDesignParameter(name={self.name}, min_value={self.min_value}, max_value={self.max_value})"


class CategoricalDesignParameter(DesignParameter):
    def __init__(self, name: str, choices: Sequence[Any]) -> None:
        super().__init__(name)
        self.choices = choices
        self.current_value = choices[0]

    def __repr__(self) -> str:
        return f"ContinuousDesignParameter(name={self.name}, choices={self.choices})"


class VectorDesignParameter(CompositeDesignParameter):
    def __init__(self, name: str, num_elements: int, min_val: float, max_val: float) -> None:
        super().__init__(name, [ContinuousDesignParameter(f"{name}_{i}", min_val, max_val) for i in range(num_elements)])

    def to_numpy(self) -> np.ndarray:
        values = [param.current_value for param in self._parameters]
        return np.array(values)

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._parameters),)