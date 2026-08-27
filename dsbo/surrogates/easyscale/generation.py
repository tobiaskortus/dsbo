from __future__ import annotations

import torch
from typing import Any
import dataclasses
from dataclasses import dataclass

from dsbo.surrogates.easyscale.random import VirtualRankContext


@dataclass(frozen=True)
class GenerationHeader:
    job_id: str
    generation: int
    physical_rank: int
    physical_world_size: int
    assigned_ests: tuple[int, ...]
    checkpoint_path: str
    target_epoch: int
    rendezvous_host: str | None = None
    rendezvous_port: int | None = None
    rank_zero_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationPlan(GenerationHeader):
    backend: str = "nccl"


@dataclass(frozen=True)
class GenerationState(GenerationHeader):
    checkpoint_path: str | None
    contexts: tuple[VirtualRankContext, ...] = dataclasses.field(kw_only=True)


@dataclass(frozen=True)
class GenerationResult(GenerationHeader):
    throughput: float = dataclasses.field(kw_only=True)
    contexts: tuple[VirtualRankContext, ...] = dataclasses.field(kw_only=True)
    should_stop: bool = False

    @property
    def model_state(self) -> dict[str, torch.Tensor] | None:
        if self.rank_zero_state is None:
            return None
        return self.rank_zero_state.get("model_state")
