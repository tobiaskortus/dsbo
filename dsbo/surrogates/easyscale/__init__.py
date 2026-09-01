from __future__ import annotations

import torch.distributed as dist

from dsbo.surrogates.easyscale.controller import EasyScaleController
from dsbo.surrogates.easyscale.protocol import (
    ESTPlan,
    EasyScaleSchedulerConfig,
    PlanDatabase,
    Proposal as EasyScaleProposal,
    assign_virtual_ranks,
)
from dsbo.surrogates.easyscale.generation import (
    GenerationAction,
    GenerationPlan,
    GenerationState,
    GenerationResult,
)
from dsbo.surrogates.easyscale.random import (
    VirtualRankContext,
    initial_context,
    _capture_random_state,
    _restore_random_state,
    seed_everything,
)
from dsbo.surrogates.easyscale.runner import run_easyscale_generation
from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.virtual_dataset import (
    VirtualBatch,
    VirtualBatchSampler,
    VirtualDataset,
    build_virtual_dataloader,
)


__all__ = [
    "EasyScaleSchedulerConfig",
    "ESTPlan",
    "PlanDatabase",
    "EasyScaleProposal",
    "assign_virtual_ranks",
    "GenerationPlan",
    "GenerationAction",
    "GenerationState",
    "GenerationResult",
    "VirtualRankContext",
    "initial_context",
    "_capture_random_state",
    "_restore_random_state",
    "seed_everything",
    "run_easyscale_generation",
    "EasyScaleStrategy",
    "VirtualBatch",
    "VirtualDataset",
    "VirtualBatchSampler",
    "build_virtual_dataloader",
    "EasyScaleController",
]
