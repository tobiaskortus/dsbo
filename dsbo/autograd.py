from __future__ import annotations

import functools
import contextvars
from typing import Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext

import torch
import torch.autograd.forward_ad as fw

from dsbo.parameters import Candidate


@dataclass(frozen=True)
class SerializedTensor:
    primal: torch.Tensor
    tangent: torch.Tensor | None = None
    grad: torch.Tensor | None = None


_current_tangent_sampler = contextvars.ContextVar("current_tangent_sampler", default=None)


@contextmanager
def use_tangent_sampler(sampler):
    token = _current_tangent_sampler.set(sampler)
    try:
        yield
    finally:
        _current_tangent_sampler.reset(token)


def get_tangent_sampler():
    return _current_tangent_sampler.get()


class DirectionPolicy(ABC):
    @abstractmethod
    def sample(
        self,
        count: int,
        dimension: int,
        *,
        generator: torch.Generator,
        like: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def gradient(self, directional: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return (directional.unsqueeze(1) * directions).mean(dim=0)


@dataclass(frozen=True)
class TangentSamplerPartition:
    candidate: Candidate
    directions: torch.Tensor

    def sample(self, design: dict[str, Any]) -> dict[str, Any]:
        decoded_rows = [self.candidate.set_tangent(direction).values for direction in self.directions]
        sampled: dict[str, Any] = {}

        for name, value in design.items():
            if not isinstance(value, torch.Tensor):
                sampled[name] = value
                continue

            primals = []
            tangents = []
            for row in decoded_rows:
                primal, tangent = fw.unpack_dual(row[name])
                primals.append(primal)
                tangents.append(torch.zeros_like(primal) if tangent is None else tangent)

            sampled[name] = fw.make_dual(torch.stack(primals), torch.stack(tangents))

        return sampled


class RandomForwardContext:
    def __init__(self, candidate: Candidate, seed: int, policy: DirectionPolicy):
        self.candidate = candidate
        self.seed = seed
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self._generator = torch.Generator(device=self.candidate.tensor.device)
        self._generator.manual_seed(self.seed)
        self._direction_parts: list[torch.Tensor] = []

    def for_task(self, module: Any, design: dict[str, Any]) -> TangentSamplerPartition | None:
        sample_count = getattr(module, "sample_count", None)
        if sample_count is None:
            return None

        count = int(sample_count(design))
        if count <= 0:
            raise ValueError(f"{module} returned an invalid forward sample count: {count}.")

        directions = self.policy.sample(
            count,
            self.candidate.tensor.numel(),
            generator=self._generator,
            like=self.candidate.tensor,
        )
        self._direction_parts.append(directions)
        return TangentSamplerPartition(self.candidate, directions)

    @property
    def directions(self) -> torch.Tensor:
        if not self._direction_parts:
            return self.candidate.tensor.new_empty((0, self.candidate.tensor.numel()))
        return torch.cat(self._direction_parts)

    def gradient(self, directional: torch.Tensor) -> torch.Tensor:
        directions = self.directions
        if directional.shape != (directions.shape[0],):
            raise ValueError(
                f"Sample count mismatch: workflow produced {directional.shape[0]} directional losses "
                f"but the simulations sampled {directions.shape[0]} tangent rows."
            )
        return self.policy.gradient(directional, directions)


def _pack(m: dict[str, Any]) -> dict[str, Any]:
    return {k: (SerializedTensor(*fw.unpack_dual(v)) if isinstance(v, torch.Tensor) else v) for k, v in m.items()}


def _unpack(m: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(m, dict):
        return m
    return {k: (fw.make_dual(v.primal, v.tangent) if isinstance(v, SerializedTensor) and v.tangent is not None and fw._current_level != -1 else v.primal if isinstance(v, SerializedTensor) else v) for k, v in m.items()}


def train_op(func: Callable):
    if hasattr(func, '_wrapped_fn'):
        return type(func)(train_op(func._wrapped_fn))

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        level = fw._current_level
        for _ in range(level + 1):
            fw.exit_dual_level()
        try:
            with torch.enable_grad():
                return func(*args, **kwargs)
        finally:
            for _ in range(level + 1):
                fw.enter_dual_level()
    return wrapper


def autograd_context(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        is_forward_ad   = kwargs.pop("is_forward_ad", False)
        is_grad_enabled = kwargs.pop("is_grad_enabled", False)
        pack_result     = kwargs.pop("pack_result", False)
        unpack_design   = kwargs.pop("unpack_design", False)
        tangent_sampler = kwargs.pop("tangent_sampler", None)

        current_level = fw._current_level
        use_existing_forward_ad_level = is_forward_ad and current_level > -1

        assert current_level == -1 or use_existing_forward_ad_level, (
            "Forward AD level should be -1 when entering autograd_context, but got level {}"
            .format(current_level)
        )

        with torch.set_grad_enabled(is_grad_enabled):
            with fw.dual_level() if is_forward_ad and not use_existing_forward_ad_level else nullcontext():
                with use_tangent_sampler(tangent_sampler):
                    if unpack_design and "design" in kwargs:
                        kwargs["design"] = _unpack(kwargs["design"])
                    result = func(*args, **kwargs)
                    return _pack(result) if pack_result and isinstance(result, dict) else result

    return wrapper
