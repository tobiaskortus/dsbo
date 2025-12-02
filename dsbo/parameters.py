import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Callable, Optional, Sequence, Tuple, Union


class DesignParameter:
    def __init__(self, name: str) -> None:
        self.name: str = name


class CompositeDesignParameter(DesignParameter):
    def __init__(self, name: str, parameters: Sequence[DesignParameter]) -> None:
        super().__init__(name)
        self._parameters = parameters

    def __repr__(self) -> str:
        return f"CompositeDesignParameter(name={self.name}, parameters={self._parameters})"
    
    @property
    def parameters(self) -> Sequence[DesignParameter]:
        return self._parameters


class ContinuousDesignParameter(DesignParameter):
    def __init__(self, name, min_value: float, max_value: float, value: Optional[float] = 0) -> None:
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value

    def __repr__(self) -> str:
        return f"ContinuousDesignParameter(name={self.name}, min_value={self.min_value}, max_value={self.max_value})"

class CategoricalDesignParameter(DesignParameter):
    def __init__(self, name: str, choices: Sequence[Any]) -> None:
        super().__init__(name)
        self.choices = choices

    def __repr__(self) -> str:
        return f"ContinuousDesignParameter(name={self.name}, choices={self.choices})"


class VectorDesignParameter(CompositeDesignParameter):
    def __init__(self, name: str, num_elements: int, min_val: float, max_val: float) -> None:
        super().__init__(name, [ContinuousDesignParameter(f"{name}_{i}", min_val, max_val) for i in range(num_elements)])

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._parameters),)


class ValueRegistry:
    def __init__(self, registry: Optional[dict] = None) -> None:
        object.__setattr__(self, '_registry', registry if registry is not None else {})
    
    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ValueRegistry is immutable")

    @classmethod
    def sample_from(cls, design: Union[CompositeDesignParameter, DesignParameter],
                    sampler_callback: Callable[[DesignParameter], Any],
                    transform: Optional[Callable[[dict], dict]] = None) -> 'ValueRegistry':
        
        def _sample_recursive(param: DesignParameter) -> Any:
            if isinstance(param, CompositeDesignParameter):
                return {p.name: _sample_recursive(p) for p in param.parameters}
            else:
                return sampler_callback(param)
        
        sampled_values = _sample_recursive(design)

        if transform:
            sampled_values = transform(sampled_values)
        
        return cls(sampled_values)
    
    @classmethod
    def sample_batch(cls, design: Union[CompositeDesignParameter, DesignParameter],
                     sampler_callback: Callable[[DesignParameter], Any],
                     batch_size: int) -> Sequence['ValueRegistry']:
        return [cls.sample_from(design, sampler_callback) for _ in range(batch_size)]
    
    def to_dict(self, flatten: bool = True) -> dict:
        if not flatten:
            import copy
            return copy.deepcopy(self._registry)
        
        def _flatten(d: dict, parent: str = '') -> dict:
            items = []
            for k, v in d.items():
                key = f"{k}" if parent else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, key).items())
                else:
                    items.append((key, v))
            return dict(items)
        
        return _flatten(self._registry)