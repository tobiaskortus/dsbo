from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import lightning as L
import torch.distributed as dist
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.fabric.plugins.collectives.torch_collective import TorchCollective

from dsbo.surrogates.easyscale.virtual_dataset import VirtualBatch
from dsbo.surrogates.easyscale.strategy.types import PendingEST
from dsbo.surrogates.easyscale.random import (
    RandomState,
    VirtualRankContext,
    virtual_rng_scope,
)
from dsbo.surrogates.easyscale.strategy.reduction import (
    aggregate_metrics,
    assign_gradients,
    log_metrics,
    reduce_gradients,
)
from dsbo.surrogates.easyscale.strategy.validation import (
    capture_log,
    contexts_by_rank,
    extract_loss,
    flatten_gradients,
    resolve_device,
    assert_accumulate_grad_batches,
    assert_automatic_optimization,
    assert_fp32,
    assert_no_buffers,
    assert_no_lbfgs,
    assert_no_mutable_batchnorm,
    validate_virtual_batch,
)


@contextmanager
def training_rng_scope(module: L.LightningModule, device: torch.device) -> Generator[None]:
    process_rng = RandomState.capture(device)
    try:
        yield
    finally:
        process_rng.restore()
        module.zero_grad(set_to_none=True)


class EasyScaleStrategy(SingleDeviceStrategy):
    def __init__(
        self,
        virtual_world_size: int,
        assigned_ests: Sequence[int],
        contexts: Sequence[VirtualRankContext] | Mapping[int, VirtualRankContext],
        *,
        physical_rank: int = 0,
        physical_world_size: int = 1,
        device: torch.device | str | None = None,
        backend: str = "nccl",
    ) -> None:
        assigned = tuple(sorted(assigned_ests))
        context_by_rank = contexts_by_rank(contexts)
        super().__init__(device=resolve_device(device))
        self.local_rank = 0
        self.backend = backend
        self.assigned_ests = assigned

        self.global_rank = physical_rank
        self.physical_rank = physical_rank
        self.world_size = physical_world_size
        self.virtual_world_size = virtual_world_size
        self.physical_world_size = physical_world_size

        self.est_steps = 0
        self._pending_batch_size = 0
        self.collective = TorchCollective()
        self._pending: list[PendingEST] | None = None
        self.contexts = {rank: context_by_rank[rank] for rank in assigned}
        self.all_contexts: tuple[VirtualRankContext, ...] | None = None

    @property
    def is_global_zero(self) -> bool:
        return self.physical_rank == 0

    def setup(self, trainer: L.Trainer) -> None:
        super().setup(trainer)
        self._validate_training_setup(trainer)

    def batch_to_device(self, batch, device=None, dataloader_idx: int = 0):
        if not isinstance(batch, VirtualBatch):
            return super().batch_to_device(batch, device, dataloader_idx)
        return batch

    def training_step(self, batch: VirtualBatch, batch_idx: int):
        validate_virtual_batch(batch, self.assigned_ests)

        pending = []
        parameters = [parameter for parameter in self.lightning_module.parameters() if parameter.requires_grad]
        parameter_sizes = [parameter.numel() for parameter in parameters]

        with training_rng_scope(self.lightning_module, self.root_device):
            for rank, est_batch in batch.items():

                est_batch = move_data_to_device(est_batch, self.root_device)
                self.lightning_module.zero_grad(set_to_none=True)
                
                with virtual_rng_scope(self.contexts[rank], self.root_device), capture_log(self.lightning_module) as metrics:
                    loss = extract_loss(self.lightning_module.training_step(est_batch, batch_idx))

                gradients = (
                    torch.autograd.grad(loss, parameters, allow_unused=True)
                    if parameters
                    else ()
                )

                flattened, mask = flatten_gradients(gradients, parameter_sizes, self.root_device)
                pending.append(PendingEST(rank, flattened, mask, float(loss.detach().cpu()), metrics))
                self.est_steps += 1

        self._pending = pending
        self._pending_batch_size = sum(len(indices) for indices in batch.indices)
        return torch.tensor(
            sum(item.loss for item in pending) / len(pending),
            device=self.root_device,
            requires_grad=True,
        )

    def backward(self, closure_loss, optimizer, *args, **kwargs):
        assert self._pending is not None, "EasyScale backward has no EST gradients."

        self.pre_backward(closure_loss)
        closure_loss = self.precision_plugin.pre_backward(
            closure_loss,
            self.lightning_module,
        )

        reduction = (
            self._pending,
            self.virtual_world_size,
            self.physical_world_size,
            self.root_device,
        )
        averages, global_mask, averaged_loss = reduce_gradients(*reduction)
        averaged_metrics = aggregate_metrics(*reduction)
        self._pending = None

        assign_gradients(self.lightning_module, averages, global_mask)
        closure_loss.detach().copy_(averaged_loss)
        log_metrics(
            self.lightning_module,
            averaged_metrics,
            averaged_loss,
            self._pending_batch_size,
        )

        closure_loss = self.precision_plugin.post_backward(closure_loss,
            self.lightning_module,
        )
        self.post_backward(closure_loss)
        return closure_loss

    def validation_step(self, *args, **kwargs):
        return self.lightning_module.validation_step(*args, **kwargs)

    def gather_contexts(self) -> tuple[VirtualRankContext, ...]:
        local = tuple(self.contexts[rank] for rank in self.assigned_ests)
        if self.physical_world_size > 1:
            gathered: list[Any] = [None for _ in range(self.physical_world_size)]
            dist.all_gather_object(gathered, local)
            contexts = tuple(context for physical_contexts in gathered for context in physical_contexts)
        else:
            contexts = local
        contexts = tuple(sorted(contexts, key=lambda context: context.rank))

        if [context.rank for context in contexts] != list(range(self.virtual_world_size)):
            raise RuntimeError("generation did not return every EST context exactly once.")
        
        self.all_contexts = contexts
        return contexts

    def reduce(self, tensor, group=None, reduce_op="mean"):
        if self.physical_world_size == 1 or not isinstance(tensor, torch.Tensor):
            return tensor
        operation = str(reduce_op).lower()
        if operation.endswith("mean") or operation.endswith("avg"):
            result = self.collective.all_reduce(tensor.clone(), op="sum")
            result.div_(self.physical_world_size)
        else:
            result = self.collective.all_reduce(tensor.clone(), op=reduce_op)
        return result

    def all_gather(self, tensor, group=None, sync_grads=False):
        if self.physical_world_size == 1:
            return tensor.unsqueeze(0)
        gathered = [torch.empty_like(tensor) for _ in range(self.physical_world_size)]
        self.collective.all_gather(gathered, tensor)
        return torch.stack(gathered)

    def broadcast(self, obj, src=0):
        if self.physical_world_size == 1:
            return obj
        if isinstance(obj, torch.Tensor):
            return self.collective.broadcast(obj, src)
        values = [obj if self.physical_rank == src else None]
        self.collective.broadcast_object_list(values, src=src)
        return values[0]

    def barrier(self, name=None):
        if self.physical_world_size > 1:
            self.collective.barrier()

    def _validate_training_setup(self, trainer: L.Trainer) -> None:
        assert_accumulate_grad_batches(trainer.accumulate_grad_batches)
        assert_automatic_optimization(self.lightning_module)
        assert_no_mutable_batchnorm(self.lightning_module)
        assert_no_buffers(self.lightning_module)
        assert_fp32(self.lightning_module)
        assert_no_lbfgs(trainer.optimizers)
