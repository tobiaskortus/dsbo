from __future__ import annotations

import torch
import  torch.autograd.forward_ad as fw
from torch import nn

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

    def repeat(self, size: int) -> "Vector":
        return Vector(self.lower, self.upper, size, self.log)

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
                    result[name] = raw[name]
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
class ParameterAccessor:
    tensor: torch.Tensor
    parameter: torch.Tensor


class CandidateTensor(torch.Tensor):
    """Flat candidate tensor whose gradients are shared with the candidate's parameters."""
    @property
    def grad(self):
        grad = torch.Tensor.grad.__get__(self)
        if grad is not None:
            return grad
        candidate = getattr(self, "candidate", None)
        if candidate is None:
            return None
        grads = [candidate.parameters[name].grad for name in candidate.parameter_names]
        return None if any(value is None for value in grads) else torch.cat(grads)

    @grad.setter
    def grad(self, gradient):
        candidate = getattr(self, "candidate", None)
        if candidate is None:
            torch.Tensor.grad.__set__(self, gradient)
            return
        parameters = candidate.parameters
        for name, (offset, width) in candidate._index.items():
            parameters[name].grad = gradient[offset:offset + width].detach()


@dataclass(frozen=True)
class Candidate:
    space: ParameterSpace
    tensor: torch.Tensor
    _index: Dict[str, Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.space, ParameterMap):
            index: Dict[str, Tuple[int, int]] = {}
            offset = 0
            seen: Dict[int, Tuple[int, int]] = {}
            for n, p in self.space._params.items():
                if isinstance(p, Constant):
                    continue
                w = len(p.choices) if hasattr(p, 'choices') else 1
                pid = id(p)
                if pid in seen:
                    index[n] = seen[pid]
                    continue
                slot = (offset, w)
                seen[pid] = slot
                index[n] = slot
                offset += w
        else:
            index = {n: (i, 1) for i, n in enumerate(self.space.source._params)}
        object.__setattr__(self, '_index', index)
        object.__setattr__(self, '_parameters', None)
        tensor = self.tensor
        if not isinstance(tensor, CandidateTensor):
            if fw.unpack_dual(tensor).tangent is None:
                tensor = torch.Tensor._make_subclass(CandidateTensor, tensor, tensor.requires_grad)
                object.__setattr__(self, 'tensor', tensor)
        if isinstance(tensor, CandidateTensor):
            object.__setattr__(tensor, 'candidate', self)

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(self._index)

    @property
    def parameters(self) -> nn.ParameterDict:
        parameters = self._parameters
        if parameters is None:
            parameters = nn.ParameterDict({
                name: nn.Parameter(self.tensor.detach()[offset:offset + width])
                for name, (offset, width) in self._index.items()
            })
            object.__setattr__(self, '_parameters', parameters)
        return parameters

    @classmethod
    def from_values(cls, space: ParameterSpace, values: Dict[str, Any], requires_grad: bool = False) -> Candidate:
        return cls(space, space.encode(values, requires_grad))

    def __getitem__(self, key):
        if not isinstance(self.space, ParameterMap):
            p = self.space._params[key]
            if isinstance(p, Constant):
                return p.value
            offset, w = self._index[key]
            tensor = self.tensor[offset:offset + w]
            return ParameterAccessor(tensor, tensor.detach())

        return self.space.decode_tensors(self.tensor)[key]
    
    @property
    def values(self) -> Dict[str, Any]:
        if not isinstance(self.space, ParameterMap):
            def collect(spec, _parts):
                result = {}
                for name, p in spec.items():
                    parts = (*_parts, name)
                    match p:
                        case Group():
                            result.update(collect(p.params, parts))
                        case Constant():
                            result[self.space._key_fn(parts)] = p.value
                        case Vector():
                            key = self.space._key_fn(parts)
                            offset, _ = self._index[key]
                            result[key] = self.tensor[offset:offset + 1].expand(p.size)
                        case Categorical():
                            key = self.space._key_fn(parts)
                            offset, w = self._index[key]
                            result[key] = p.choices[int(self.tensor[offset:offset + w].argmax().item())]
                        case Continuous():
                            key = self.space._key_fn(parts)
                            offset, _ = self._index[key]
                            result[key] = self.tensor[offset]
                        case Integer():
                            key = self.space._key_fn(parts)
                            offset, _ = self._index[key]
                            result[key] = int(round(self.tensor[offset].item()))
                return result
            return collect(self.space._spec, ())
        return self.space._decode_values(self.tensor)
    

    def detach(self) -> Candidate:
        return Candidate(self.space, self.tensor.detach().clone())

    def requires_grad_(self, requires_grad: bool = True) -> Candidate:
        self.tensor.requires_grad_(requires_grad)
        return self

    @property
    def grad(self) -> Optional[torch.Tensor]:
        return self.tensor.grad

    def sample_tangent(self) -> Candidate:
        primal, _ = fw.unpack_dual(self.tensor)
        return Candidate(self.space, fw.make_dual(primal, torch.randn_like(primal)))
    
    def set_tangent(self, tangent: torch.Tensor) -> Candidate:
        assert self.tensor.shape == tangent.shape, "Tangent must have the same shape as the primal tensor"
        
        primal, _ = fw.unpack_dual(self.tensor)
        return Candidate(self.space, fw.make_dual(primal, tangent))
    
    def __contains__(self, key: str) -> bool:
        params = (self.space.target._params \
                    if isinstance(self.space, ParameterMap) \
                    else self.space._params)
        return key in params
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented
        return self.values == other.values


class CandidateBatch(list[Candidate]):
    def __init__(self, candidates: list[Candidate]):
        super().__init__(candidates)

    def append(self, object):
        self._changed = True
        return super().append(object)
    
    def detach(self) -> CandidateBatch:
        return CandidateBatch([c.detach() for c in self])

    def requires_grad_(self, requires_grad: bool = True) -> CandidateBatch:
        for c in self: c.requires_grad_(requires_grad)
        return self
    
    def sample_tangent(self) -> CandidateBatch:
        return CandidateBatch([c.sample_tangent() for c in self])
    
    def set_tangent(self, tangent: torch.Tensor) -> CandidateBatch:
        assert tangent.ndim == 2, "Tangent must have the shape N_elements x N_parameters"
        return CandidateBatch([c.set_tangent(t) for c, t in zip(self, tangent)])
    
    @property
    def tensor(self) -> torch.Tensor:
        return torch.stack([c.tensor for c in self])

    @property
    def grad(self) -> torch.Tensor:
        return torch.stack([c.grad for c in self])

    @property
    def space(self):
        return self[0].space
    
    @property
    def _index(self):
        return self[0]._index


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
        self._params, self._aliases = _collect(spec, key_fn)
        self._canonical_names: Dict[str, List[str]] = {}
        for key in self._params:
            self._canonical_names.setdefault(key, []).append(key)
        for exposed, canonical in self._aliases.items():
            self._canonical_names.setdefault(canonical, []).append(exposed)
        self._sampler: Sampler | None = None

    @staticmethod
    def repeat(prefix: str, n: int, **params: Union[DesignParameter, Group]) -> Dict[str, Union[DesignParameter, Group]]:
        group = Group(**params)
        return {f"{prefix}_{i}": group for i in range(n)}
    
    def register_sampler(self, sampler: Sampler) -> None:
        sampler.bind(self)
        self._sampler = sampler

    def encode(self, values: Dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        seen: set[int] = set()
        for name, p in self:
            if isinstance(p, Constant):
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            src = next((s for s in self._canonical_names.get(name, [name]) if s in values), name)
            match p:
                case Categorical():
                    v = values[src]
                    one_hot = [0.0] * len(p.choices)
                    one_hot[p.choices.index(v)] = 1.0
                    parts.append(torch.tensor(one_hot, dtype=torch.float32))
                case Vector():
                    v = values[src]
                    if isinstance(v, torch.Tensor) and v.numel() != 1:
                        raise ValueError(
                            f"Vector parameter '{name}' expects a single value (homogeneous), "
                            f"got {v.numel()} elements."
                        )
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v, dtype=torch.float32)
                    parts.append(v.reshape(1))
                case _:
                    v = values[src]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v, dtype=torch.float32)
                    parts.append(v.reshape(1))
        encoded = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
        if requires_grad:
            encoded.requires_grad_(requires_grad)
        else:
            encoded = encoded.detach()
        return encoded

    def sample(self, 
               sampler: Sampler = None, 
               requires_grad: bool = False,
               sample_tangent: bool = False) -> Candidate:
        
        assert not (requires_grad and sample_tangent)

        s = sampler or self._sampler
        if s is None:
            raise RuntimeError("No sampler provided. Pass one to sample() or call register_sampler() first.")
        
        #TODO: this is hacky
        s.bind(self)
        t = s.ask()

        if requires_grad:
            t.requires_grad_(True)
        if sample_tangent:
            t = fw.make_dual(t, torch.rand_like(t))
        
        return Candidate(self, t)

    def sample_batch(self, sampler: Sampler = None, batch_size: int = 16, requires_grad: bool = False) -> CandidateBatch:
        return CandidateBatch([self.sample(sampler, requires_grad) for _ in range(batch_size)])

    def dims(self, encoding: Encoding | str = Encoding.ORDINAL, include_constants: bool = False) -> int:
        seen: set[int] = set()
        unique: list[DesignParameter] = []
        for p in self._params.values():
            if id(p) in seen:
                continue
            seen.add(id(p))
            unique.append(p)
        match encoding:
            case Encoding.ORDINAL | "ordinal":
                return sum(0 if isinstance(p, Constant) and not include_constants else 1 for p in unique)
            case Encoding.ONE_HOT | "one_hot":
                return sum(
                    0 if isinstance(p, Constant) and not include_constants else
                    len(p.choices) if isinstance(p, Categorical) else 1
                    for p in unique
                )
            case _:
                raise ValueError(f"Unknown encoding {encoding!r}. Use 'ordinal' or 'one_hot'.")
            
    def __iter__(self)  -> Iterator[Tuple[str, DesignParameter]]:
        return iter(self._params.items())


class CandidateBatch(list[Candidate]):
    def __init__(self, candidates: list[Candidate]):
        super().__init__(candidates)
        self._space = candidates[0].space

    @property
    def space(self) -> ParameterSpace:
        return self[0].space

    @property
    def tensor(self) -> torch.Tensor:
        return torch.stack([c.tensor for c in self])

    @property
    def grad(self) -> Optional[torch.Tensor]:
        if any(c.grad is None for c in self):
            raise ValueError("Not all candidates have gradients")
        return torch.stack([c.grad for c in self])

    def detach(self) -> CandidateBatch:
        return CandidateBatch([c.detach() for c in self])

    def requires_grad_(self, requires_grad: bool = True) -> CandidateBatch:
        for c in self:
            c.requires_grad_(requires_grad)
        return self

    def sample_tangent(self) -> CandidateBatch:
        return CandidateBatch([c.sample_tangent() for c in self])

    def set_tangent(self, tangent: torch.Tensor) -> CandidateBatch:
        assert tangent.shape[0] == len(self), "Tangent batch size must match candidate batch size"
        for i, c in enumerate(self):
            c.set_tangent(tangent[i])
        return self



def _collect(params: Dict[str, Union[DesignParameter, Group]],
             key_fn: Callable[[Tuple[str, ...]], str] = '.'.join,
             _parts: Tuple[str, ...] = (),
             _seen: Optional[Dict[int, str]] = None,
             _aliases: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, DesignParameter], Dict[str, str]]:
    """Flatten spec into {exposed_key: parameter}, detecting shared parameters.

    Reusing the same parameter object under more than one key shares it: the
    later keys are recorded as aliases of the first (canonical) key in the
    returned alias map, so the shared parameter occupies a single latent
    dimension and fans out on decode.
    """
    out: Dict[str, Union[Continuous, Integer, Categorical, Vector]] = {}
    if _seen is None:
        _seen = {}
    if _aliases is None:
        _aliases = {}
    for name, spec in params.items():
        parts = (*_parts, name)
        if isinstance(spec, Group):
            sub, _ = _collect(spec.params, key_fn, parts, _seen, _aliases)
            out.update(sub)
        else:
            key = key_fn(parts)
            pid = id(spec)
            if pid in _seen:
                _aliases[key] = _seen[pid]
            else:
                _seen[pid] = key
            out[key] = spec
    return out, _aliases