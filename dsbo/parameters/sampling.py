from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import Optional


class Sampler(ABC):
    @abstractmethod
    def ask(self, n: int, dims: int) -> torch.Tensor:
        """Return n design points in the unit hypercube [0, 1]^dims as a tensor of shape (n, dims)."""
        pass


class UniformSampler(Sampler):
    def __init__(self, seed: Optional[int] = None):
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def ask(self, n: int, dims: int) -> torch.Tensor:
        return torch.rand(n, dims, generator=self._generator)


class SobolSampler(Sampler):
    def __init__(self, seed: Optional[int] = None, scramble: bool = True):
        self._seed = seed
        self._scramble = scramble
        self._engine: Optional[torch.quasirandom.SobolEngine] = None
        self._dims: Optional[int] = None

    def ask(self, n: int, dims: int) -> torch.Tensor:
        if self._engine is None or self._dims != dims:
            self._engine = torch.quasirandom.SobolEngine(dimension=dims, scramble=self._scramble, seed=self._seed)
            self._dims = dims
        return self._engine.draw(n)


class LatinHypercubeSampler(Sampler):
    def __init__(self, seed: Optional[int] = None):
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def ask(self, n: int, dims: int) -> torch.Tensor:
        strata = torch.stack([torch.randperm(n, generator=self._generator) for _ in range(dims)], dim=1)
        offsets = torch.rand(n, dims, generator=self._generator)
        return (strata.float() + offsets) / n
