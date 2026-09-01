from __future__ import annotations

import torch
import torch.distributed as dist

from dsbo.surrogates.easyscale.strategy.types import PendingEST


def _coerce_pending(pending: list[PendingEST]) -> list[PendingEST]:
    return [
        item if isinstance(item, PendingEST) else PendingEST(*item)
        for item in pending
    ]


def reduce_gradients(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
):
    pending = _coerce_pending(pending)
    gradient_size = pending[0].gradient.numel()
    parameter_count = pending[0].mask.numel()
    losses = torch.zeros(virtual_world_size, dtype=torch.float32, device=device)
    ownership = torch.zeros(virtual_world_size, dtype=torch.float32, device=device)
    masks = torch.zeros((virtual_world_size, parameter_count), dtype=torch.float32, device=device)
    contributions = torch.zeros((virtual_world_size, gradient_size), dtype=torch.float32, device=device)

    for item in pending:
        ownership[item.rank] += 1
        masks[item.rank].copy_(item.mask.to(dtype=torch.float32))
        losses[item.rank] = item.loss
        contributions[item.rank].copy_(item.gradient)

    metadata = torch.cat((ownership, masks.reshape(-1), losses))
    if physical_world_size > 1:
        dist.all_reduce(metadata, op=dist.ReduceOp.SUM)
        dist.all_reduce(contributions, op=dist.ReduceOp.SUM)

    head = virtual_world_size
    tail = head + virtual_world_size * parameter_count
    ownership = metadata[:head]
    masks = metadata[head:tail].reshape(virtual_world_size, parameter_count)
    losses = metadata[tail:]

    try:
        if ownership.tolist() != [1.0] * virtual_world_size:
            raise RuntimeError("gradient reduction requires every EST exactly once.")
        ordered_masks = masks.to(dtype=torch.bool)
        if any(not torch.equal(mask, ordered_masks[0]) for mask in ordered_masks[1:]):
            raise RuntimeError("a parameter gradient is missing from only some ESTs.")
        return (
            contributions.div(virtual_world_size).sum(dim=0),
            ordered_masks[0].clone(),
            losses.mean(),
        )
    except Exception as exc:
        raise RuntimeError(f"EasyScale gradient reduction failed: {type(exc).__name__}: {exc}") from exc


def aggregate_metrics(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
) -> dict[str, tuple]:
    pending = _coerce_pending(pending)
    schema = None
    names = set()
    for item in pending:
        names.update(item.metrics)
        if schema is None:
            schema = item.metrics

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
            schema[name][1],
            dict(schema[name][2]),
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


def log_metrics(module, averaged_metrics, averaged_loss, batch_size: int) -> None:
    averaged_metrics = dict(averaged_metrics)
    averaged_metrics.setdefault(
        "train_loss",
        (
            averaged_loss.detach().item(),
            (),
            {"on_step": False, "on_epoch": True, "prog_bar": True},
        ),
    )
    for name, (value, args, kwargs) in averaged_metrics.items():
        kwargs = dict(kwargs, sync_dist=False)
        if kwargs.get("on_epoch"):
            kwargs["batch_size"] = batch_size
        module.log(name, value, *args, **kwargs)
