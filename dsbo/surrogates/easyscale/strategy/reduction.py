from __future__ import annotations

import time

import torch
import torch.distributed as dist

from dsbo.surrogates.easyscale.strategy.types import PendingEST


def _coerce_pending(pending: list[PendingEST]) -> list[PendingEST]:
    return [
        item if isinstance(item, PendingEST) else PendingEST(*item)
        for item in pending
    ]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_all_reduce(
    tensor: torch.Tensor,
    device: torch.device,
    timings: dict[str, float] | None,
    name: str,
) -> None:
    if timings is None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return
    _synchronize(device)
    started = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    _synchronize(device)
    if timings is not None:
        timings[name] = time.perf_counter() - started


def reduce_gradients(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
    timings: dict[str, float] | None = None,
):
    started = time.perf_counter()
    pending = _coerce_pending(pending)
    gradient_size = pending[0].gradient.numel()
    parameter_count = pending[0].mask.numel()
    losses = torch.zeros(virtual_world_size, dtype=torch.float32, device=device)
    ownership = torch.zeros(virtual_world_size, dtype=torch.float32, device=device)
    masks = torch.zeros((virtual_world_size, parameter_count), dtype=torch.float32, device=device)

    for item in pending:
        ownership[item.rank] += 1
        masks[item.rank].copy_(item.mask.to(dtype=torch.float32))
        losses[item.rank] = item.loss

    metadata = torch.cat((ownership, masks.reshape(-1), losses))
    if physical_world_size > 1:
        _timed_all_reduce(
            metadata, device, timings, "gradient_metadata_all_reduce"
        )
    elif timings is not None:
        timings["gradient_metadata_all_reduce"] = 0.0

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

        contributions = torch.zeros((virtual_world_size, gradient_size), dtype=torch.float32, device=device)
        if physical_world_size > 1:
            slots = (virtual_world_size + physical_world_size - 1) // physical_world_size
            if len(pending) > slots:
                raise RuntimeError("gradient gather requires balanced EST assignments.")
            local = torch.zeros((slots, gradient_size + 1), dtype=torch.float32, device=device)
            local[:, 0] = -1
            for index, item in enumerate(pending):
                local[index, 0] = item.rank
                local[index, 1:].copy_(item.gradient)

            gathered = torch.empty(
                (physical_world_size * slots, gradient_size + 1),
                dtype=torch.float32,
                device=device,
            )
            if timings is None:
                dist.all_gather_into_tensor(gathered, local)
            else:
                _synchronize(device)
                gather_started = time.perf_counter()
                dist.all_gather_into_tensor(gathered, local)
                _synchronize(device)
                timings["gradient_tensor_all_gather"] = (
                    time.perf_counter() - gather_started
                )

            gathered_ranks = gathered[:, 0].to(dtype=torch.int64).tolist()
            sorthed_gathered_ranks = sorted(rank for rank in gathered_ranks if rank >= 0)
            if sorthed_gathered_ranks != list(range(virtual_world_size)):
                raise RuntimeError("gradient gather requires every EST exactly once.")
            for index, rank in enumerate(gathered_ranks):
                if rank >= 0:
                    contributions[rank].copy_(gathered[index, 1:])
        else:
            if timings is not None:
                timings["gradient_tensor_all_gather"] = 0.0
            for item in pending:
                contributions[item.rank].copy_(item.gradient)

        result = (
            contributions.div(virtual_world_size).sum(dim=0),
            ordered_masks[0].clone(),
            losses.mean(),
        )
        if timings is not None:
            timings["gradient_aggregation_total"] = time.perf_counter() - started
        return result
    except Exception as exc:
        raise RuntimeError(f"EasyScale gradient reduction failed: {type(exc).__name__}: {exc}") from exc


def aggregate_metrics(
    pending: list[PendingEST],
    virtual_world_size: int,
    physical_world_size: int,
    device: torch.device,
    timings: dict[str, float] | None = None,
) -> dict[str, tuple]:
    started = time.perf_counter()
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
        _timed_all_reduce(packed, device, timings, "metric_all_reduce")
    elif timings is not None:
        timings["metric_all_reduce"] = 0.0

    split = values.numel()
    values = packed[:split].reshape_as(values)
    presence = packed[split:].reshape_as(presence)

    if not bool(presence.to(dtype=torch.bool).all()):
        raise RuntimeError("training metrics require every EST exactly once.")
    
    result = {
        name: (
            float((values[:, index].sum() / virtual_world_size).item()),
            schema[name][1],
            dict(schema[name][2]),
        )
        for index, name in enumerate(names)
    }
    if timings is not None:
        timings["metric_aggregation_total"] = time.perf_counter() - started
    return result


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
