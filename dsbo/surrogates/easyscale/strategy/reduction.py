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


def _timed_all_reduce(tensor: torch.Tensor, device: torch.device, timings: dict[str, float] | None, name: str) -> None:
    if timings is None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return
    
    _synchronize(device)
    started = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    _synchronize(device)
    if timings is not None:
        timings[name] = time.perf_counter() - started


def _timed_barrier(device: torch.device, timings: dict[str, float] | None, name: str) -> None:
    if timings is None:
        return
    _synchronize(device)
    started = time.perf_counter()
    dist.barrier()
    _synchronize(device)
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
        _timed_barrier(device, timings, "gradient_pre_collective_wait")
        _timed_all_reduce(metadata, device, timings, "gradient_metadata_all_reduce")
    elif timings is not None:
        timings["gradient_pre_collective_wait"] = 0.0
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

        slots = (virtual_world_size + physical_world_size - 1) // physical_world_size
        if len(pending) > slots:
            raise RuntimeError("gradient reduction requires balanced EST assignments.")

        if physical_world_size > 1:
            local_ranks = torch.full((slots,), -1, dtype=torch.int64, device=device)
            for index, item in enumerate(pending):
                local_ranks[index] = item.rank
            gathered_ranks = torch.empty(
                physical_world_size * slots,
                dtype=torch.int64,
                device=device,
            )
            if timings is not None:
                _synchronize(device)
            rank_gather_started = time.perf_counter()
            dist.all_gather_into_tensor(gathered_ranks, local_ranks)
            if timings is not None:
                _synchronize(device)
                timings["gradient_rank_all_gather"] = (
                    time.perf_counter() - rank_gather_started
                )

            rank_rows = gathered_ranks.reshape(physical_world_size, slots).tolist()
            if sorted(rank for rows in rank_rows for rank in rows if rank >= 0) != list(
                range(virtual_world_size)
            ):
                raise RuntimeError("gradient exchange requires every EST exactly once.")

            if gradient_size:
                chunk_size = (gradient_size + physical_world_size - 1) // physical_world_size
                send_shards = torch.zeros(
                    (physical_world_size, len(pending), chunk_size),
                    dtype=torch.float32,
                    device=device,
                )
                for index, item in enumerate(pending):
                    for peer in range(physical_world_size):
                        start = peer * chunk_size
                        end = min(start + chunk_size, gradient_size)
                        if start < end:
                            send_shards[peer, index, : end - start].copy_(item.gradient[start:end])

                physical_rank = dist.get_rank()
                received = [
                    send_shards[peer]
                    if peer == physical_rank
                    else torch.empty(
                        (sum(rank >= 0 for rank in rows), chunk_size),
                        dtype=torch.float32,
                        device=device,
                    )
                    for peer, rows in enumerate(rank_rows)
                ]
                operations = []
                for peer in range(physical_world_size):
                    if peer == physical_rank:
                        continue

                    operations.append(dist.P2POp(dist.isend, send_shards[peer], peer))
                    operations.append(dist.P2POp(dist.irecv, received[peer], peer))

                if timings is not None:
                    _synchronize(device)
                exchange_started = time.perf_counter()
                requests = dist.batch_isend_irecv(operations)
                for request in requests:
                    request.wait()
                if timings is not None:
                    _synchronize(device)
                    timings["gradient_shard_exchange"] = (time.perf_counter() - exchange_started)

                ordered_gradients: list[torch.Tensor | None] = [None] * virtual_world_size
                for rows, values in zip(rank_rows, received):
                    for rank, value in zip((rank for rank in rows if rank >= 0), values):
                        ordered_gradients[rank] = value
                local_average = torch.zeros(chunk_size, dtype=torch.float32, device=device)

                for gradient in ordered_gradients:
                    if gradient is None:
                        raise RuntimeError("gradient exchange omitted an EST.")
                    local_average.add_(gradient, alpha=1 / virtual_world_size)

                _timed_barrier(device, timings, "gradient_result_wait")
                gathered_average = torch.empty(physical_world_size * chunk_size, dtype=torch.float32, device=device)
                if timings is not None:
                    _synchronize(device)
                result_gather_started = time.perf_counter()
                dist.all_gather_into_tensor(gathered_average, local_average)
                if timings is not None:
                    _synchronize(device)
                    timings["gradient_result_all_gather"] = (time.perf_counter() - result_gather_started)
                averages = gathered_average[:gradient_size]
            else:
                averages = torch.empty(0, dtype=torch.float32, device=device)
                if timings is not None:
                    timings["gradient_shard_exchange"] = 0.0
                    timings["gradient_result_wait"] = 0.0
                    timings["gradient_result_all_gather"] = 0.0
        else:
            if timings is not None:
                timings["gradient_rank_all_gather"] = 0.0
                timings["gradient_shard_exchange"] = 0.0
                timings["gradient_result_wait"] = 0.0
                timings["gradient_result_all_gather"] = 0.0
            ordered_gradients = [None] * virtual_world_size
            for item in pending:
                ordered_gradients[item.rank] = item.gradient
            averages = torch.zeros(gradient_size, dtype=torch.float32, device=device)
            for gradient in ordered_gradients:
                if gradient is None:
                    raise RuntimeError("gradient reduction omitted an EST.")
                averages.add_(gradient, alpha=1 / virtual_world_size)

        result = (
            averages,
            ordered_masks[0].clone(),
            losses.mean(),
        )
        if timings is not None:
            timings["gradient_collective_total"] = sum(
                timings.get(name, 0.0)
                for name in (
                    "gradient_metadata_all_reduce",
                    "gradient_rank_all_gather",
                    "gradient_shard_exchange",
                    "gradient_result_all_gather",
                )
            )
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
