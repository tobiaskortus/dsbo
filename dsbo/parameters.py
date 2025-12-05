import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Callable, Generator, Optional, Sequence, Tuple, Union


class DesignParameter:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def iterate_flatten(self) -> Generator['DesignParameter', None, None]:
        yield self
    
    def num_parameters(self, one_hot_categorical: bool = True) -> int:
        return 1


class CompositeDesignParameter(DesignParameter):
    def __init__(self, name: str, parameters: Sequence[DesignParameter]) -> None:
        super().__init__(name)
        self._parameters = parameters

    def iterate_flatten(self) -> Generator['DesignParameter', None, None]:
        for param in self._parameters:
            if isinstance(param, CompositeDesignParameter):
                yield from param.iterate_flatten()
            else:
                yield param

    def num_parameters(self, one_hot_categorical: bool = True) -> int:
        return sum(p.num_parameters(one_hot_categorical) for p in self._parameters)

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

    def num_parameters(self, one_hot_categorical: bool = True) -> int:
        return len(self.choices) if one_hot_categorical else 1

    def __repr__(self) -> str:
        return f"ContinuousDesignParameter(name={self.name}, choices={self.choices})"


class VectorDesignParameter(CompositeDesignParameter):
    def __init__(self, name: str, num_elements: int, min_val: float, max_val: float) -> None:
        super().__init__(name, [ContinuousDesignParameter(f"{name}_{i}", min_val, max_val) for i in range(num_elements)])

    @property
    def shape(self) -> Tuple[int]:
        return (len(self._parameters),)


class ValueRegistry:
    def __init__(self, registry: Optional[dict] = None, design: Optional[Union[CompositeDesignParameter, DesignParameter]] = None) -> None:
        object.__setattr__(self, '_registry', registry if registry is not None else {})
        object.__setattr__(self, '_design', design)
    
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
        
        return cls(sampled_values, design)
    
    @classmethod
    def sample_batch(cls, design: Union[CompositeDesignParameter, DesignParameter],
                     sampler_callback: Callable[[DesignParameter], Any],
                     batch_size: int) -> Sequence['ValueRegistry']:
        return [cls.sample_from(design, sampler_callback) for _ in range(batch_size)]
    
    @classmethod
    def from_design_vector(cls, design: Union[CompositeDesignParameter, DesignParameter],
                           design_vector: Sequence[float],
                           one_hot_categorical: bool = True) -> 'ValueRegistry':

        design_vector = list(design_vector)
        index = 0
        
        def _assign_recursive(param: DesignParameter) -> Any:
            nonlocal index
            
            if isinstance(param, CompositeDesignParameter):
                return {p.name: _assign_recursive(p) for p in param.parameters}
            elif isinstance(param, CategoricalDesignParameter) and one_hot_categorical:
                num_choices = len(param.choices)
                one_hot = design_vector[index:index + num_choices]
                index += num_choices
                
                if len(one_hot) < num_choices:
                    raise ValueError(f"Insufficient vector length for one-hot encoding of {param.name}")
                
                choice_idx = int(np.argmax(one_hot))
                return param.choices[choice_idx]
            else:
                value = design_vector[index]
                index += 1
                return value
        
        structured_values = _assign_recursive(design)
        
        if index != len(design_vector):
            raise ValueError(f"Design vector length mismatch: expected {index} elements, got {len(design_vector)}")
        
        return cls(structured_values, design)
    

    def to_design_vector(self, one_hot_categorical: bool = True) -> list:
        def _flatten(param: Optional[DesignParameter], value: Any) -> list:
            if isinstance(value, dict):
                result = []
                if param and isinstance(param, CompositeDesignParameter):
                    for p in param.parameters:
                        result.extend(_flatten(p, value[p.name]))
                else:
                    for v in value.values():
                        result.extend(_flatten(None, v))
                return result
            
            if one_hot_categorical and param and isinstance(param, CategoricalDesignParameter):
                one_hot = [0.0] * len(param.choices)
                choice_idx = param.choices.index(value)
                one_hot[choice_idx] = 1.0
                return one_hot
            
            return [value]
        
        return _flatten(self._design, self._registry)
    

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