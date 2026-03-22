from __future__ import annotations

import torch
from enum import Enum
from attr import field
from numpy.char import lower, upper
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union, TypeAlias

class Encoding(Enum):
    ORDINAL = "ordinal"
    ONE_HOT = "one_hot"

#TODO: add option to declare Variable constants (e.g. fixed hyperparameters) that are not included in the encoding/tensors.
#TODO: add support for default values? alternative to sampling?

@dataclass(frozen=True)
class Continuous:
    lower: float
    upper: float
    log: bool = False

@dataclass(frozen=True)
class Integer:
    lower: int
    upper: int


@dataclass(frozen=True)
class Categorical:
    choices: List[Any]


@dataclass(frozen=True)
class Vector:
    lower: float
    upper: float
    size: int
    log: bool = False


@dataclass(frozen=True)
class Constant:
    value: Union[float, int, str]


DesignParameter: TypeAlias = Union[Continuous, Integer, Categorical, Vector, Constant]


class Group:
    def __init__(self, **params: Union[DesignParameter, Group]):
        self.params = params


class ParameterMap(ABC):
    def __init__(self, source: ParameterSpace, target: ParameterSpace):
        self.source = source
        self.target = target

    @abstractmethod
    def decode_tensors(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Differentiable latent -> target decode. Must preserve the autograd graph."""
        pass

    def _assert_decoded_keys(self, decoded: Dict[str, torch.Tensor]) -> None:
        expected_keys = {k for k, p in self.target._params.items() if not isinstance(p, Constant)}
        decoded_keys = set(decoded.keys())
        missing = expected_keys - decoded_keys
        extra = decoded_keys - expected_keys
        
        if missing:
            raise ValueError(f"decode_tensors output missing keys: {missing}")
        if extra:
            raise ValueError(f"decode_tensors output has unexpected keys: {extra}")
    
    def _decode_values(self, z: torch.Tensor) -> Dict[str, Any]:
        raw = self.decode_tensors(z)
        self._assert_decoded_keys(raw)

        result: Dict[str, Any] = {}
        for name, p in self.target._params.items():
            match p:
                case Constant():
                    result[name] = p.value
                    continue
                #TODO: fix this if gradient is required (especially categorical and integer + .item())
                case Categorical():
                    t = raw[name]
                    expected_w = len(p.choices)
                    if t.shape[-1] != expected_w:
                        raise ValueError(
                            f"decode_tensors output for '{name}' has width {t.shape[-1]}, "
                            f"expected {expected_w} (Categorical with {expected_w} choices)"
                        )
                    result[name] = p.choices[int(t.argmax().item())]
                case Integer():
                    result[name] = int(raw[name].item())
                case Continuous():
                    result[name] = float(raw[name].item())
        return result

    def register_sampler(self, sampler: Sampler) -> None:
        self.source.register_sampler(sampler)

    def __call__(self, source_candidate: Candidate) -> Candidate:
        return Candidate(self, source_candidate.tensor)

    def sample(self, sampler: Sampler = None, requires_grad: bool = False) -> Candidate:
        return Candidate(self, self.source.sample(sampler, requires_grad).tensor)

    def n_dims(self, encoding: str = 'ordinal') -> int:
        return self.target.n_dims(encoding)


@dataclass(frozen=True)
class Candidate:
    space: ParameterSpace
    tensor: torch.Tensor
    _index: Dict[str, Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.space, ParameterMap):
            index: Dict[str, Tuple[int, int]] = {}
            offset = 0
            for n, p in self.space._params.items():
                if isinstance(p, Constant):
                    continue
                w = len(p.choices) if hasattr(p, 'choices') else 1
                index[n] = (offset, w)
                offset += w
        else:
            index = {n: (i, 1) for i, n in enumerate(self.space.source._params)}
        object.__setattr__(self, '_index', index)

    @classmethod
    def from_values(cls, space: ParameterSpace, values: Dict[str, Any], requires_grad: bool = False) -> Candidate:
        return cls(space, space.encode(values, requires_grad))

    def __getitem__(self, key):
        if not isinstance(self.space, ParameterMap):
            p = self.space._params[key]
            if isinstance(p, Constant):
                return p.value
            offset, w = self._index[key]
            return self.tensor[offset:offset+w] if w > 1 else self.tensor[offset].item()
        
        return self.space.decode_tensors(self.tensor)[key]
    
    @property
    def values(self) -> Dict[str, Any]:
        if not isinstance(self.space, ParameterMap):
            result: Dict[str, Any] = {}
            for name, p in self.space._params.items():
                match p:
                    case Constant():
                        result[name] = p.value
                    case Categorical():   # Categorical — argmax over one-hot slice
                        offset, w = self._index[name]
                        result[name] = p.choices[int(self.tensor[offset:offset + w].argmax().item())]
                    case Continuous():
                        offset, _ = self._index[name]
                        result[name] = float(self.tensor[offset].item())
                    case Integer():
                        offset, _ = self._index[name]
                        result[name] = int(round(self.tensor[offset].item()))
                    case _:
                        raise ValueError(f"Unsupported parameter type: {type(p)}")
            return result
        return self.space._decode_values(self.tensor)
    
    def __contains__(self, key: str) -> bool:
        params = (self.space.target._params \
                    if isinstance(self.space, ParameterMap) \
                    else self.space._params)
        return key in params
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.values == other.values


class Sampler(ABC):
    def bind(self, space: ParameterSpace) -> None:
        self._space = space

    @abstractmethod
    def ask(self) -> torch.Tensor: 
        pass


class ParameterSpace:
    def __init__(self, *, key_fn: Callable[[Tuple[str, ...]], str] = lambda x: ".".join(x),
                 **spec: Union[DesignParameter, Group]):
        
        self._key_fn = key_fn
        self._spec = spec
        self._params = _collect(spec, key_fn)
        self._sampler: Sampler | None = None

    @staticmethod
    def repeat(prefix: str, n: int, **params: Union[DesignParameter, Group]) -> Dict[str, Union[DesignParameter, Group]]:
        return {f"{prefix}_{i}": Group(**params) for i in range(n)}
    
    def register_sampler(self, sampler: Sampler) -> None:
        sampler.bind(self)
        self._sampler = sampler

    def encode(self, values: Dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        parts: List[float] = []
        for name, p in self:
            match p:
                case Constant():
                    continue
                case Categorical():
                    v = values[name]
                    one_hot = [0.0] * len(p.choices)
                    one_hot[p.choices.index(v)] = 1.0
                    parts.extend(one_hot)
                case _:
                    parts.append(float(values[name]))
        return torch.tensor(parts, dtype=torch.float32, requires_grad=requires_grad)

    def sample(self, sampler: Sampler = None, requires_grad: bool = False) -> Candidate:
        s = sampler or self._sampler
        if s is None:
            raise RuntimeError("No sampler provided. Pass one to sample() or call register_sampler() first.")
        
        #TODO: this is hacky
        s.bind(self)
        t = s.ask()
        if requires_grad:
            t.requires_grad_(True)
        return Candidate(self, t)

    def sample_batch(self, sampler: Sampler = None, batch_size: int = 16, requires_grad: bool = False) -> list[Candidate]:
        return [self.sample(sampler, requires_grad) for _ in range(batch_size)]

    def dims(self, encoding: Encoding | str = Encoding.ORDINAL, include_constants: bool = False) -> int:
        match encoding:
            case Encoding.ORDINAL | "ordinal":
                return sum(0 if isinstance(p, Constant) and not include_constants else 1 for p in self._params.values())
            case Encoding.ONE_HOT | "one_hot":
                return sum(
                    0 if isinstance(p, Constant) and not include_constants else
                    len(p.choices) if isinstance(p, Categorical) else 1
                    for p in self._params.values()
                )
            case _:
                raise ValueError(f"Unknown encoding {encoding!r}. Use 'ordinal' or 'one_hot'.")
            
    def __iter__(self)  -> Iterator[Tuple[str, DesignParameter]]:
        return iter(self._params.items())


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
                out[key_fn((*_parts, f"{name}_{i}"))] = Continuous(spec.lower, spec.upper, spec.log)
        else:
            out[key_fn(parts)] = spec
    return out