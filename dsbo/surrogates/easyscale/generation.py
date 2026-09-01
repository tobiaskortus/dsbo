from __future__ import annotations

import torch
from typing import Any
import dataclasses
from dataclasses import dataclass
from enum import IntEnum

from dsbo.surrogates.easyscale.random import VirtualRankContext


class GenerationAction(IntEnum):
    CONTINUE = 0
    RECONFIGURE = 1
    STOP = 2
    FAIL = 3


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
class GenerationState:
    checkpoint_path: str | None
    completed_epoch: int
    contexts: tuple[VirtualRankContext, ...]
    rank_zero_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationResult(GenerationHeader):
    throughput: float = dataclasses.field(kw_only=True)
    contexts: tuple[VirtualRankContext, ...] = dataclasses.field(kw_only=True)
    should_stop: bool = False
    action: GenerationAction = GenerationAction.STOP
    timings: dict[str, float] = dataclasses.field(default_factory=dict, kw_only=True)

    @property
    def model_state(self) -> dict[str, torch.Tensor] | None:
        if self.rank_zero_state is None:
            return None
        return self.rank_zero_state.get("model_state")
