from __future__ import annotations

import torch
import torch.autograd.forward_ad as fw
from torch import nn

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from dsbo.parameters.types import (Constant, DesignParameter, Encoding, Group,
                                   Vector, _collect, _collect_vectors)
from dsbo.parameters.sampling import Sampler


class _ParameterCollection(dict):
    """Flattened parameter map that also exposes vector parameters by their base key."""

    def __init__(self, params: Dict[str, DesignParameter], vectors: Dict[str, Vector]):
        super().__init__(params)
        self._vectors = vectors

    def __getitem__(self, key: str) -> DesignParameter:
        if key in self:
            return super().__getitem__(key)
        if key in self._vectors:
            return self._vectors[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or key in self._vectors

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class ParameterSpace:
    def __init__(self, *, key_fn: Callable[[Tuple[str, ...]], str] = lambda x: ".".join(x),
                 **spec: Union[DesignParameter, Group]):
        
        self._key_fn = key_fn
        self._spec = spec
        vectors = {key: vector for key, (vector, _) in _collect_vectors(spec, key_fn).items()}
        self._params = _ParameterCollection(_collect(spec, key_fn), vectors)
        self._index: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for name, parameter in self._params.items():
            width = parameter.width()
            if width == 0:
                continue
            self._index[name] = (offset, width)
            offset += width
        self._sampler: Sampler | None = None

    @staticmethod
    def repeat(prefix: str, n: int, **params: Union[DesignParameter, Group]) -> Dict[str, Union[DesignParameter, Group]]:
        return {f"{prefix}_{i}": Group(**params) for i in range(n)}

    def __iter__(self) -> Iterator[Tuple[str, DesignParameter]]:
        return iter(self._params.items())

    def dims(self, encoding: Encoding | str = Encoding.ORDINAL, include_constants: bool = False) -> int:
        if encoding not in (Encoding.ORDINAL, Encoding.ONE_HOT, "ordinal", "one_hot"):
            raise ValueError(f"Unknown encoding {encoding!r}. Use 'ordinal' or 'one_hot'.")
        return sum(p.dims(encoding, include_constants) for p in self._params.values())

    def encode(self, values: Dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        # Pre-expand any independent vector values (values[name] = array) into individual keys.
        expanded = dict(values)
        for base_key, (vector, flat_keys) in _collect_vectors(self._spec, self._key_fn).items():
            if vector.shared:
                continue
            if base_key in values:
                for flat_key, v in zip(flat_keys, values[base_key]):
                    expanded[flat_key] = v

        parts: List[float] = []
        for name, p in self:
            if isinstance(p, Constant):
                continue
            parts.extend(p.encode(expanded[name]))
        return torch.tensor(parts, dtype=torch.float32, requires_grad=requires_grad)

    def unit_to_design_tensor(self, u: torch.Tensor) -> torch.Tensor:
        """Map points from the unit hypercube [0, 1]^n_latent to the encoded tensor layout."""
        if u.ndim == 1:
            u = u.unsqueeze(0)

        blocks: List[torch.Tensor] = []
        offset = 0
        for p in self._params.values():
            if isinstance(p, Constant):
                continue
            blocks.append(p.from_unit(u[:, offset:offset + 1]))
            offset += 1

        if not blocks:
            return torch.empty(u.shape[0], 0, dtype=torch.float32)
        return torch.cat(blocks, dim=1).to(torch.float32)

    def register_sampler(self, sampler: Sampler) -> None:
        self._sampler = sampler

    def sample(self, sampler: Sampler = None, requires_grad: bool = False) -> Candidate:
        s = sampler or self._sampler
        if s is None:
            raise RuntimeError("No sampler provided. Pass one to sample() or call register_sampler() first.")
        
        t = self.unit_to_design_tensor(s.ask(1, self.dims(Encoding.ORDINAL)))[0]

        if requires_grad:
            t.requires_grad_(True)
        
        return Candidate(self, t)

    def sample_batch(self, sampler: Sampler = None, batch_size: int = 16, requires_grad: bool = False) -> CandidateBatch:
        s = sampler or self._sampler
        if s is None:
            raise RuntimeError("No sampler provided. Pass one to sample_batch() or call register_sampler() first.")
        
        tensors = self.unit_to_design_tensor(s.ask(batch_size, self.dims(Encoding.ORDINAL)))
        return CandidateBatch([
            Candidate(self, tensors[i].detach().requires_grad_(requires_grad))
            for i in range(batch_size)
        ])


class CandidateTensor(torch.Tensor):
    """Flat candidate tensor whose gradients are shared with the candidate's parameters."""

    @property
    def grad(self):
        grad = torch.Tensor.grad.__get__(self) if self.is_leaf else None
        if grad is not None:
            return grad
        candidate = getattr(self, "candidate", None)
        if candidate is None:
            return None
        grads = [candidate.parameter_dict[name].grad for name in candidate.parameter_names]
        return None if any(value is None for value in grads) else torch.cat(grads)

    @grad.setter
    def grad(self, gradient):
        candidate = getattr(self, "candidate", None)
        if candidate is None:
            torch.Tensor.grad.__set__(self, gradient)
            return
        parameters = candidate.parameter_dict
        for name, (offset, width) in candidate._index.items():
            parameters[name].grad = gradient[offset:offset + width].detach()


@dataclass(frozen=True)
class Candidate:
    space: ParameterSpace
    tensor: torch.Tensor
    _index: Dict[str, Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, '_index', self.space._index)
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
    def parameter_dict(self) -> Dict[str, nn.Parameter]:
        parameters = self._parameters
        if parameters is None:
            parameters = {
                name: nn.Parameter(self.tensor.detach()[offset:offset + width])
                for name, (offset, width) in self._index.items()
            }
            object.__setattr__(self, '_parameters', parameters)
        return parameters

    def parameters(self) -> Iterator[Tuple[str, DesignParameter, torch.Tensor]]:
        for name, (offset, width) in self._index.items():
            yield name, self.space._params[name], self.tensor[offset:offset + width]

    @classmethod
    def from_values(cls, space: ParameterSpace, values: Dict[str, Any], requires_grad: bool = False) -> Candidate:
        return cls(space, space.encode(values, requires_grad))

    @property
    def values(self) -> Dict[str, Any]:
        def collect(spec, _parts):
            result = {}
            for name, p in spec.items():
                parts = (*_parts, name)
                match p:
                    case Group():
                        result.update(collect(p.params, parts))
                    case Constant():
                        result[self.space._key_fn(parts)] = p.decode()
                    case Vector():
                        key = self.space._key_fn(parts)
                        lookup = key if p.shared else self.space._key_fn((*_parts, f"{name}_1"))
                        offset, _ = self._index[lookup]
                        result[key] = p.decode(self.tensor[offset:offset + p.width()])
                    case _:
                        key = self.space._key_fn(parts)
                        offset, w = self._index[key]
                        result[key] = p.decode(self.tensor[offset] if w == 1 else self.tensor[offset:offset + w])
            return result
        return collect(self.space._spec, ())

    def detach(self) -> Candidate:
        return type(self)(self.space, self.tensor.detach().clone())

    def requires_grad_(self, requires_grad: bool = True) -> Candidate:
        self.tensor.requires_grad_(requires_grad)
        return self

    @property
    def grad(self) -> Optional[torch.Tensor]:
        return self.tensor.grad

    def set_tangent(self, tangent: torch.Tensor) -> Candidate:
        assert self.tensor.shape == tangent.shape, "Tangent must have the same shape as the primal tensor"
        
        primal, _ = fw.unpack_dual(self.tensor)
        return type(self)(self.space, fw.make_dual(primal, tangent))

    def __getitem__(self, key):
        p = self.space._params[key]
        if isinstance(p, Constant):
            return p.value
        offset, w = self._index[key]
        return self.tensor[offset:offset+w] if w > 1 else self.tensor[offset].item()
    
    def __contains__(self, key: str) -> bool:
        return key in self.space._params

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return NotImplemented

        def eq(a: Any, b: Any) -> bool:
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                        and a.shape == b.shape and torch.equal(a.detach(), b.detach()))
            return a == b

        left, right = self.values, other.values
        return left.keys() == right.keys() and all(eq(v, right[k]) for k, v in left.items())

    def __repr__(self) -> str:
        parts = []
        for key, value in self.values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().tolist()
            parts.append(f"{key}={value}")
        return f"{type(self).__name__}({', '.join(parts)})"


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

    def set_tangent(self, tangent: torch.Tensor) -> CandidateBatch:
        assert tangent.shape[0] == len(self), "Tangent batch size must match candidate batch size"
        return CandidateBatch([c.set_tangent(t) for c, t in zip(self, tangent)])
