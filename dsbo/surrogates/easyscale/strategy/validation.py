from __future__ import annotations

import lightning as L
import torch
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from torch.nn.modules.batchnorm import _BatchNorm
from typing import Any

from dsbo.surrogates.easyscale.random import VirtualRankContext
from dsbo.surrogates.easyscale.virtual_dataset import VirtualBatch


def resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    return torch.device(device)


def coerce_metric_value(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("EasyScale only supports scalar training metrics.")
        return float(value.detach().cpu())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError("EasyScale only supports numeric scalar training metrics.")


@contextmanager
def capture_log(module: L.LightningModule) -> Generator[dict[str, tuple]]:
    metrics: dict[str, tuple] = {}
    original_log = module.log
    original_log_dict = module.log_dict

    def capture_metric(name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        metrics[name] = (coerce_metric_value(value), args, dict(kwargs))

    def capture_metrics(values: Mapping[str, Any], *args: Any, **kwargs: Any) -> None:
        for name, value in values.items():
            capture_metric(name, value, *args, **kwargs)

    module.log = capture_metric
    module.log_dict = capture_metrics
    try:
        yield metrics
    finally:
        module.log = original_log
        module.log_dict = original_log_dict


def extract_loss(output: Any) -> torch.Tensor:
    loss = output.get("loss") if isinstance(output, Mapping) else output
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError("EasyScale training_step must return one scalar loss tensor.")
    return loss


def validate_virtual_batch(batch: object, assigned_ests: Sequence[int]) -> None:
    if not isinstance(batch, VirtualBatch):
        raise TypeError("EasyScale training requires a VirtualBatch.")
    if tuple(sorted(batch.ranks)) != assigned_ests:
        raise RuntimeError("VirtualBatch does not match this worker's assigned ESTs.")


def flatten_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameter_sizes: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = torch.zeros(sum(parameter_sizes), device=device, dtype=torch.float32)
    mask = torch.zeros(len(parameter_sizes), device=device, dtype=torch.bool)
    offset = 0
    for index, (gradient, size) in enumerate(zip(gradients, parameter_sizes)):
        if gradient is not None:
            flattened[offset : offset + size].copy_(gradient.detach().reshape(-1))
            mask[index] = True
        offset += size
    return flattened, mask


def contexts_by_rank(contexts: Sequence[VirtualRankContext] | Mapping[int, VirtualRankContext]) -> dict[int, VirtualRankContext]:
    return (
        dict(contexts)
        if isinstance(contexts, Mapping)
        else {context.rank: context for context in contexts}
    )


def assert_accumulate_grad_batches(accumulate_grad_batches: int) -> None:
    if accumulate_grad_batches != 1:
        raise ValueError("EasyScale requires accumulate_grad_batches=1.")


def assert_automatic_optimization(module: L.LightningModule) -> None:
    if not module.automatic_optimization:
        raise ValueError("EasyScale requires Lightning automatic optimization.")


def assert_no_mutable_batchnorm(module: L.LightningModule) -> None:
    if any(
        isinstance(module, _BatchNorm) and module.track_running_stats
        for module in module.modules()
    ):
        raise ValueError("EasyScale does not support mutable BatchNorm state.")


def assert_no_buffers(module: L.LightningModule) -> None:
    if any(True for _ in module.buffers()):
        raise ValueError("EasyScale does not support registered model buffers.")


def assert_fp32(module: L.LightningModule) -> None:
    if any(
        tensor.is_floating_point() and tensor.dtype != torch.float32
        for tensor in (*module.parameters(), *module.buffers())
    ):
        raise ValueError("EasyScale requires FP32 parameters and buffers.")


def assert_no_lbfgs(optimizers: Sequence[Any]) -> None:
    if any(isinstance(optimizer, torch.optim.LBFGS) for optimizer in optimizers):
        raise ValueError("EasyScale does not support closure-reentrant LBFGS.")
