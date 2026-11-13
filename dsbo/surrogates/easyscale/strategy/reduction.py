from __future__ import annotations

import torch
import torch.distributed as dist

from dsbo.surrogates.easyscale.strategy.types import PendingEST


def reduce_gradients(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
):
    gradient_size = pending[0].gradient.numel()
    parameter_count = pending[0].mask.numel()
    head = virtual_world_size
    tail = head + virtual_world_size * parameter_count
    metadata = torch.zeros(tail + virtual_world_size, dtype=torch.float32, device=device)

    for item in pending:
        metadata[item.rank] += 1
        metadata[head + item.rank * parameter_count : head + (item.rank + 1) * parameter_count].copy_(item.mask)
        metadata[tail + item.rank] = item.loss

    if physical_world_size > 1:
        dist.all_reduce(metadata, op=dist.ReduceOp.SUM)

    ownership = metadata[:head]
    masks = metadata[head:tail].reshape(virtual_world_size, parameter_count)
    losses = metadata[tail:]

    try:
        if ownership.tolist() != [1.0] * virtual_world_size:
            raise RuntimeError("gradient reduction requires every EST exactly once.")
        
        ordered_masks = masks.to(dtype=torch.bool)
        if any(not torch.equal(mask, ordered_masks[0]) for mask in ordered_masks[1:]):
            raise RuntimeError("a parameter gradient is missing from only some ESTs.")

        slots = (virtual_world_size + physical_world_size - 1) // physical_world_size
        if len(pending) > slots:
            raise RuntimeError("gradient reduction requires balanced EST assignments.")

        averages = pending[0].gradient
        for item in pending[1:]:
            averages.add_(item.gradient)
        if physical_world_size > 1 and gradient_size:
            dist.all_reduce(averages, op=dist.ReduceOp.SUM)
        averages.div_(virtual_world_size)
        return (averages, ordered_masks[0].clone(), losses.mean())
    
    except Exception as exc:
        raise RuntimeError(f"EasyScale gradient reduction failed: {type(exc).__name__}: {exc}") from exc


def aggregate_metrics(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
) -> dict[str, tuple]:
    schema: dict[str, tuple] | None = None
    names: set[str] = set()
    for item in pending:
        signature = {
            name: (tuple(value[1]), dict(value[2]))
            for name, value in item.metrics.items()
        }
        names.update(signature)
        if schema is None:
            schema = signature
        elif signature != schema:
            raise RuntimeError("training metrics must expose an identical schema on every EST.")

    if not names:
        return {}

    names = sorted(names)
    values = torch.zeros((virtual_world_size, len(names)), dtype=torch.float32, device=device)
    presence = torch.zeros_like(values)

    for item in pending:
        for index, name in enumerate(names):
            if name in item.metrics:
                values[item.rank, index] = item.metrics[name][0]
                presence[item.rank, index] = 1

    packed = torch.cat((values.reshape(-1), presence.reshape(-1)))
    if physical_world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    split = values.numel()
    values = packed[:split].reshape_as(values)
    presence = packed[split:].reshape_as(presence)

    if not bool(presence.to(dtype=torch.bool).all()):
        raise RuntimeError("training metrics require every EST exactly once.")

    return {
        name: (
            float((values[:, index].sum() / virtual_world_size).item()),
            schema[name][0],
            dict(schema[name][1]),
        )
        for index, name in enumerate(names)
    }


def assign_gradients(module, averages, global_mask) -> None:
    offset = 0
    for index, parameter in enumerate(parameter for parameter in module.parameters() if parameter.requires_grad):
        end = offset + parameter.numel()
        with torch.no_grad():
            parameter.grad = (
                averages[offset:end].view_as(parameter).clone()
                if global_mask[index]
                else None
            )
        offset = end
