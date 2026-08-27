from __future__ import annotations

import copy
import socket
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import timedelta

import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping

from dsbo.elastic import EasyScaleSchedulerConfig
from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.virtual_dataset import build_virtual_dataloader
from dsbo.surrogates.easyscale.checkpoint import EasyScaleCheckpointCallback
from dsbo.surrogates.easyscale.generation import (
    GenerationPlan,
    GenerationResult,
    GenerationState,
)
from dsbo.surrogates.easyscale.random import (
    RandomState,
    _capture_random_state,
    _restore_random_state,
    seed_everything,
)
def connect_dask():
    try:
        from distributed import get_client, get_worker

        return get_client(), get_worker()
    except ValueError:
        return None, None


def allocation_timeout(training_config) -> float:
    return float(
        getattr(
            getattr(training_config, "easyscale_scheduler", None),
            "allocation_timeout",
            300.0,
        )
    )


@contextmanager
def framework_state_preserved(device: torch.device):
    saved = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    rng_state = RandomState.capture(device)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        rng_state.restore()
        torch.use_deterministic_algorithms(saved["deterministic"])
        torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]


@contextmanager
def process_group(plan: GenerationPlan, client, worker, timeout: float):
    rendezvous_host = plan.rendezvous_host
    rendezvous_port = plan.rendezvous_port
    rendezvous_variable = None
    started = False
    try:
        if dist.is_initialized():
            raise RuntimeError("EasyScale generation found an existing process group.")
        if plan.physical_world_size > 1 and rendezvous_host is None:
            if client is None:
                raise RuntimeError(
                    "multi-worker EasyScale rendezvous requires a Dask client."
                )
            from distributed import Variable

            rendezvous_variable = Variable(
                f"dsbo-easyscale-rendezvous-{plan.job_id}-{plan.generation}",
                client=client,
            )
        if plan.physical_rank == 0:
            rendezvous_host = (
                rendezvous_host
                or (getattr(worker, "host", None) if worker is not None else None)
                or socket.gethostbyname(socket.gethostname())
            )
            store = dist.TCPStore(
                rendezvous_host,
                rendezvous_port or 0,
                plan.physical_world_size,
                True,
                timedelta(seconds=timeout),
                wait_for_workers=False,
            )
            rendezvous_port = store.port
            if rendezvous_variable is not None:
                rendezvous_variable.set(
                    (rendezvous_host, rendezvous_port),
                    timeout=timeout,
                )
        else:
            if rendezvous_host is None or rendezvous_port is None:
                rendezvous_host, rendezvous_port = rendezvous_variable.get(
                    timeout=timeout
                )
            store = dist.TCPStore(
                rendezvous_host,
                int(rendezvous_port),
                plan.physical_world_size,
                False,
                timedelta(seconds=timeout),
            )
        dist.init_process_group(
            backend=plan.backend,
            store=store,
            rank=plan.physical_rank,
            world_size=plan.physical_world_size,
            timeout=timedelta(seconds=timeout),
        )
        started = True
        yield rendezvous_host, rendezvous_port
    finally:
        try:
            if started and dist.is_initialized():
                dist.destroy_process_group()
        finally:
            if plan.physical_rank == 0 and rendezvous_variable is not None:
                rendezvous_variable.delete()


def _run_generation_body(
    plan: GenerationPlan,
    state: GenerationState,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    training_config,
    external_callbacks: Sequence[L.Callback],
    rendezvous_host: str | None,
    rendezvous_port: int | None,
    device: torch.device,
) -> GenerationResult:
    scheduler_config = getattr(
        training_config,
        "easyscale_scheduler",
        EasyScaleSchedulerConfig(),
    )
    seed = int(getattr(training_config, "seed", 0))
    virtual_world_size = len(state.contexts)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    framework_state = state.rank_zero_state or plan.rank_zero_state or {}
    if framework_state:
        _restore_random_state(framework_state, device)
    else:
        seed_everything(seed, device)

    context_by_rank = {context.rank: context for context in state.contexts}
    assigned_contexts = {
        rank: context_by_rank[rank] for rank in plan.assigned_ests
    }
    strategy = EasyScaleStrategy(
        virtual_world_size=virtual_world_size,
        assigned_ests=plan.assigned_ests,
        contexts=assigned_contexts,
        physical_rank=plan.physical_rank,
        physical_world_size=plan.physical_world_size,
        device=device,
        backend=plan.backend,
        allow_cpu=scheduler_config.allow_cpu,
    )
    virtual_loader = build_virtual_dataloader(
        train_loader,
        virtual_world_size,
        seed,
        assigned_ranks=plan.assigned_ests,
        contexts=assigned_contexts,
    )
    checkpoint_callback = EasyScaleCheckpointCallback(
        strategy,
        plan.checkpoint_path,
        rank_zero_state=state.rank_zero_state,
    )
    callbacks: list[L.Callback] = copy.deepcopy(list(external_callbacks))
    patience = getattr(training_config, "patience", None)
    if patience is not None:
        if val_loader is None:
            raise ValueError("EasyScale patience requires a validation loader.")
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        )
    callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        max_epochs=plan.target_epoch,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        strategy=strategy,
        gradient_clip_val=getattr(training_config, "grad_clip", None),
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        precision="32-true",
    )
    training_started = time.perf_counter()
    trainer.fit(
        model,
        train_dataloaders=virtual_loader,
        val_dataloaders=val_loader,
        ckpt_path=state.checkpoint_path,
    )
    training_elapsed = max(
        time.perf_counter() - training_started,
        torch.finfo(torch.float64).eps,
    )
    if not checkpoint_callback.saved:
        checkpoint_callback.save(trainer)

    contexts = strategy.all_contexts
    if contexts is None:
        raise RuntimeError("EasyScale generation did not gather EST contexts.")
    completed = torch.tensor(
        [strategy.est_steps, training_elapsed],
        dtype=torch.float64,
        device=device,
    )
    if plan.physical_world_size > 1:
        dist.all_reduce(completed[:1], op=dist.ReduceOp.SUM)
        dist.all_reduce(completed[1:], op=dist.ReduceOp.MAX)

    rank_zero_state = None
    if strategy.is_global_zero:
        rank_zero_state = {
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            **_capture_random_state(device),
        }
    if not strategy.is_global_zero:
        contexts = tuple(assigned_contexts.values())

    return GenerationResult(
        job_id=plan.job_id,
        generation=plan.generation,
        physical_rank=plan.physical_rank,
        physical_world_size=plan.physical_world_size,
        assigned_ests=plan.assigned_ests,
        checkpoint_path=plan.checkpoint_path,
        target_epoch=plan.target_epoch,
        throughput=float(completed[0].item() / completed[1].item()),
        contexts=contexts,
        should_stop=bool(trainer.should_stop),
        rendezvous_host=rendezvous_host,
        rendezvous_port=rendezvous_port,
        rank_zero_state=rank_zero_state,
    )


def run_easyscale_generation(
    plan: GenerationPlan,
    state: GenerationState,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    training_config,
    model_tag: str | None = None,
    external_callbacks: Sequence[L.Callback] = (),
) -> GenerationResult:
    client, worker = connect_dask()
    timeout = allocation_timeout(training_config)
    if plan.backend == "nccl" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "NCCL EasyScale generation requires exactly one visible GPU."
        )
    device = (
        torch.device("cuda", 0)
        if plan.backend == "nccl"
        else torch.device("cpu")
    )

    with framework_state_preserved(device):
        with process_group(plan, client, worker, timeout) as (
            rendezvous_host,
            rendezvous_port,
        ):
            return _run_generation_body(
                plan,
                state,
                model,
                train_loader,
                val_loader,
                training_config,
                external_callbacks,
                rendezvous_host,
                rendezvous_port,
                device,
            )
