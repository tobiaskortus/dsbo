from __future__ import annotations

import torch
import torch.autograd.forward_ad as fw

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from dsbo.parameters.types import (Categorical, Constant, DesignParameter, Encoding, Group,
                                   Vector, _collect, _collect_vectors)
from dsbo.parameters.sampling import Sampler


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

    def __iter__(self) -> Iterator[Tuple[str, DesignParameter]]:
        return iter(self._params.items())

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

    def encode(self, values: Dict[str, Any], requires_grad: bool = False) -> torch.Tensor:
        # Pre-expand any vector values (values[name] = array) into individual keys
        expanded = dict(values)
        for base_key, (_, flat_keys) in _collect_vectors(self._spec, self._key_fn).items():
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


@dataclass(frozen=True)
class Candidate:
    space: ParameterSpace
    tensor: torch.Tensor
    _index: Dict[str, Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        index: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for n, p in self.space._params.items():
            if isinstance(p, Constant):
                continue
            w = len(p.choices) if hasattr(p, 'choices') else 1
            index[n] = (offset, w)
            offset += w
        object.__setattr__(self, '_index', index)

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
                        offset, _ = self._index[self.space._key_fn((*_parts, f"{name}_1"))]
                        result[self.space._key_fn(parts)] = p.decode(self.tensor[offset:offset + p.size])
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
        return self.values == other.values

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
