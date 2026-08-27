from __future__ import annotations

import torch.distributed as dist

from dsbo.elastic import (
    AllocationVector,
    EasyScaleSchedulerConfig,
    ESTPlan,
    PlanDatabase,
    Proposal as EasyScaleProposal,
    ResourceVector,
    FrozenCapabilities,
    _normalize_int_vector,
    assign_virtual_ranks,
)
from dsbo.surrogates.easyscale.generation import (
    GenerationPlan,
    GenerationState,
    GenerationResult,
)
from dsbo.surrogates.easyscale.random import (
    VirtualRankContext,
    _initial_context,
    _capture_random_state,
    _restore_random_state,
    seed_everything,
)
from dsbo.surrogates.easyscale.runner import _run_easyscale_generation
from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.virtual_dataset import (
    VirtualBatch,
    VirtualDataset,
    VirtualBatchSampler,
    build_virtual_dataloader,
)


def __getattr__(name: str):
    if name == "EasyScaleController":
        from dsbo.elastic.controller import ElasticController

        return ElasticController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EasyScaleController",
]
