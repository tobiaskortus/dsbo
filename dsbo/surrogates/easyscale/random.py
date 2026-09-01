from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

import numpy as np
import torch


@dataclass
class VirtualRankContext:
    rank: int
    random_state: RandomState
    sampler_epoch: int = 0
    sampler_position: int = 0


def initial_context(
    rank: int,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> VirtualRankContext:
    rank_seed = seed + rank
    generator = torch.Generator().manual_seed(rank_seed)
    return VirtualRankContext(
        rank=rank,
        random_state=RandomState(
            device=device,
            python_rng=random.Random(rank_seed).getstate(),
            numpy_rng=np.random.RandomState(rank_seed).get_state(),
            torch_rng=generator.get_state(),
            cuda_rng=None,
            seed=rank_seed,
        ),
    )


def seed_everything(seed: int, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)


@dataclass
class RandomState:
    device: torch.device
    python_rng: object
    numpy_rng: tuple[Any, ...]
    torch_rng: torch.Tensor
    cuda_rng: torch.Tensor | None = None
    seed: int | None = None

    @classmethod
    def capture(cls, device: torch.device) -> "RandomState":
        return cls(
            device=device,
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            ),
        )

    def restore(self) -> None:
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.torch_rng)
        if self.device.type == "cuda":
            if self.cuda_rng is not None:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            elif self.seed is not None:
                torch.cuda.manual_seed(self.seed)


@contextmanager
def seed_scope(seed: int, device: torch.device) -> Generator[None]:
    state = RandomState.capture(device)
    seed_everything(seed, device)
    try:
        yield
    finally:
        state.restore()


@contextmanager
def virtual_rng_scope(context: VirtualRankContext, device: torch.device) -> Generator[None]:
    context.random_state.device = device
    context.random_state.restore()
    try:
        yield
    finally:
        context.random_state = RandomState.capture(device)


def _capture_random_state(device: torch.device) -> dict[str, Any]:
    state = RandomState.capture(device)
    return {
        "python_rng": state.python_rng,
        "numpy_rng": state.numpy_rng,
        "torch_rng": state.torch_rng,
        "cuda_rng": state.cuda_rng,
    }


def _restore_random_state(state, device: torch.device) -> None:
    RandomState(
        device=device,
        python_rng=state["python_rng"],
        numpy_rng=state["numpy_rng"],
        torch_rng=state["torch_rng"],
        cuda_rng=state.get("cuda_rng"),
    ).restore()
