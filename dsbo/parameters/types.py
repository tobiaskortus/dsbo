from __future__ import annotations

import torch
import torch.nn.functional as F

from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union, TypeAlias


class Encoding(Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one_hot"

#TODO: add support for default values? alternative to sampling?

@dataclass(frozen=True)
class Continuous:
    lower: float
    upper: float

    def encode(self, value: Any) -> List[float]:
        return [value]

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        return self.lower + u * (self.upper - self.lower)

    def decode(self, t: torch.Tensor) -> torch.Tensor:
        return t

@dataclass(frozen=True)
class Integer:
    lower: int
    upper: int

    def encode(self, value: Any) -> List[float]:
        return [value]

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        span = self.upper - self.lower + 1
        return self.lower + (u * span).floor().clamp(max=span - 1)

    def decode(self, t: torch.Tensor) -> int:
        return int(round(t.item()))


@dataclass(frozen=True)
class Categorical:
    choices: List[Any]

    def encode(self, value: Any) -> List[float]:
        one_hot = [0.0] * len(self.choices)
        one_hot[self.choices.index(value)] = 1.0
        return one_hot

    def from_unit(self, u: torch.Tensor) -> torch.Tensor:
        k = len(self.choices)
        idx = (u[:, 0] * k).long().clamp(max=k - 1)
        return F.one_hot(idx, num_classes=k).to(torch.float32)

    def decode(self, t: torch.Tensor) -> Any:
        return self.choices[int(t.argmax())]


@dataclass(frozen=True)
class Vector:
    lower: float
    upper: float
    size: int

    def decode(self, t: torch.Tensor) -> torch.Tensor:
        return t


@dataclass(frozen=True)
class Constant:
    value: Union[float, int, str]

    def decode(self) -> Union[float, int, str]:
        return self.value


DesignParameter: TypeAlias = Union[Continuous, Integer, Categorical, Vector, Constant]


class Group:
    def __init__(self, **params: Union[DesignParameter, Group]):
        self.params = params


def _collect(params: Dict[str, Union[DesignParameter, Group]], 
             key_fn: Callable[[Tuple[str, ...]], str] = '.'.join,
            _parts: Tuple[str, ...] = ()) -> Dict[str, DesignParameter]:
    """Flatten dictionary representation of parameters {key: parameter}; groups and vectors are expanded"""
    out: Dict[str, Union[Continuous, Integer, Categorical]] = {}
    for name, spec in params.items():
        parts = (*_parts, name)
        if isinstance(spec, Group):
            out.update(_collect(spec.params, key_fn, parts))
        elif isinstance(spec, Vector):
            for i in range(1, spec.size + 1):
                out[key_fn((*_parts, f"{name}_{i}"))] = Continuous(spec.lower, spec.upper)
        else:
            out[key_fn(parts)] = spec
    return out


def _collect_vectors(params: Dict[str, Union[DesignParameter, Group]],
                     key_fn: Callable[[Tuple[str, ...]], str] = '.'.join,
                     _parts: Tuple[str, ...] = ()) -> Dict[str, Tuple[Vector, List[str]]]:
    """Returns {base_key: (Vector, [flat_keys])} for all Vector params, recursing into Groups."""
    out: Dict[str, Tuple[Vector, List[str]]] = {}
    for name, spec in params.items():
        parts = (*_parts, name)
        if isinstance(spec, Group):
            out.update(_collect_vectors(spec.params, key_fn, parts))
        elif isinstance(spec, Vector):
            flat_keys = [key_fn((*_parts, f"{name}_{i}")) for i in range(1, spec.size + 1)]
            out[key_fn(parts)] = (spec, flat_keys)
    return out
