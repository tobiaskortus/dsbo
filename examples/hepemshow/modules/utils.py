from typing import TYPE_CHECKING

import torch

from dsbo.utils.process import available_cores

if TYPE_CHECKING:
    from dsbo.worker.spec import WorkerSpec


def seed_for_replica(seed: int, replica: int | None) -> int:
    return seed + (replica or 0) * 1_000_003


def resolve_num_workers(num_workers: int | None, resources: "WorkerSpec | None") -> int:
    return max(1, num_workers) if num_workers is not None else available_cores(resources)


def backpropagate_design_gradients(design, adjoint):
    inputs, gradients = [], []
    for key in ("absorber", "gap"):
        tensor = design[key]
        if tensor.requires_grad:
            gradient = torch.as_tensor(adjoint[key], device=tensor.device, dtype=tensor.dtype)
            inputs.append(tensor)
            gradients.append(gradient.expand_as(tensor))
    if inputs:
        torch.autograd.backward(inputs, gradients)
