from __future__ import annotations

import copy
import dataclasses
import socket
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import cloudpickle
import lightning as L
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping

from dsbo.logger import WorkflowTaskLogger
from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.virtual_dataset import (
    VirtualDataLoader,
    build_virtual_dataloader,
)
from dsbo.surrogates.easyscale.scheduler.plugin import EASYSCALE_RUNTIME_PLUGIN_NAME
from dsbo.surrogates.easyscale.callbacks import (
    EasyScaleCheckpointCallback,
    EasyScaleControlCallback,
)
from dsbo.surrogates.easyscale.generation import (
    GenerationAction,
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
    from distributed import get_client, get_worker

    try:
        client = get_client()
    except ValueError:
        client = None
    try:
        worker = get_worker()
    except ValueError:
        worker = None
    return client, worker


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
def process_group(
    plan: GenerationPlan,
    client,
    worker,
    timeout: float,
    rendezvous_store: dist.TCPStore | None = None,
):
    timings = {}
    rendezvous_started = time.perf_counter()
    WorkflowTaskLogger.get_current().log(
        f"easyscale rendezvous start generation={plan.generation} "
        f"rank={plan.physical_rank} world={plan.physical_world_size} "
        f"backend={plan.backend} timeout={timeout:.1f}s"
    )
    rendezvous_host = plan.rendezvous_host
    rendezvous_port = plan.rendezvous_port
    rendezvous_variable = None
    started = False
    try:
        if dist.is_initialized():
            raise RuntimeError("EasyScale generation found an existing process group.")
        if plan.physical_world_size == 1:
            timings.update(
                endpoint=0.0,
                store=0.0,
                rank_alignment=0.0,
                process_group_init=0.0,
                rendezvous=0.0,
            )
            yield rendezvous_host, rendezvous_port, timings
            return
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
        endpoint_started = time.perf_counter()
        if plan.physical_rank == 0:
            rendezvous_host = (
                rendezvous_host
                or (getattr(worker, "host", None) if worker is not None else None)
                or socket.gethostbyname(socket.gethostname())
            )
            store = rendezvous_store or dist.TCPStore(
                rendezvous_host,
                rendezvous_port or 0,
                plan.physical_world_size,
                True,
                timedelta(seconds=timeout),
                wait_for_workers=False,
            )
            timings["store"] = time.perf_counter() - endpoint_started
            rendezvous_port = store.port
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous store-ready generation={plan.generation} "
                f"rank={plan.physical_rank} host={rendezvous_host} "
                f"port={rendezvous_port} elapsed={timings['store']:.3f}s"
            )
            if rendezvous_variable is not None:
                publish_started = time.perf_counter()
                rendezvous_variable.set(
                    (rendezvous_host, rendezvous_port),
                    timeout=timeout,
                )
                timings["endpoint"] = time.perf_counter() - publish_started
                WorkflowTaskLogger.get_current().log(
                    f"easyscale rendezvous endpoint-published generation={plan.generation} "
                    f"rank={plan.physical_rank} elapsed={timings['endpoint']:.3f}s"
                )
        else:
            if rendezvous_host is None or rendezvous_port is None:
                endpoint_started = time.perf_counter()
                WorkflowTaskLogger.get_current().log(
                    f"easyscale rendezvous endpoint-wait generation={plan.generation} "
                    f"rank={plan.physical_rank}"
                )
                rendezvous_host, rendezvous_port = rendezvous_variable.get(
                    timeout=timeout
                )
                timings["endpoint"] = time.perf_counter() - endpoint_started
                WorkflowTaskLogger.get_current().log(
                    f"easyscale rendezvous endpoint-ready generation={plan.generation} "
                    f"rank={plan.physical_rank} host={rendezvous_host} "
                    f"port={rendezvous_port} elapsed={timings['endpoint']:.3f}s"
                )
            store_started = time.perf_counter()
            store = dist.TCPStore(
                rendezvous_host,
                int(rendezvous_port),
                plan.physical_world_size,
                False,
                timedelta(seconds=timeout),
            )
            timings["store"] = time.perf_counter() - store_started
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous store-ready generation={plan.generation} "
                f"rank={plan.physical_rank} host={rendezvous_host} "
                f"port={rendezvous_port} elapsed={timings['store']:.3f}s"
            )
        rank_alignment_started = time.perf_counter()
        store.set(f"rank-{plan.physical_rank}-ready", "1")
        WorkflowTaskLogger.get_current().log(
            f"easyscale rendezvous rank-ready generation={plan.generation} "
            f"rank={plan.physical_rank}"
        )
        if plan.physical_rank == 0:
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous rank-alignment-wait generation={plan.generation} "
                f"rank={plan.physical_rank} expected={plan.physical_world_size}"
            )
            store.wait(
                [f"rank-{rank}-ready" for rank in range(plan.physical_world_size)],
                timedelta(seconds=timeout),
            )
            store.set("ranks-ready", "1")
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous ranks-aligned generation={plan.generation} "
                f"rank={plan.physical_rank}"
            )
        else:
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous rank-alignment-wait generation={plan.generation} "
                f"rank={plan.physical_rank}"
            )
            store.wait(["ranks-ready"], timedelta(seconds=timeout))
        timings["rank_alignment"] = time.perf_counter() - rank_alignment_started
        WorkflowTaskLogger.get_current().log(
            f"easyscale rendezvous rank-alignment-done generation={plan.generation} "
            f"rank={plan.physical_rank} elapsed={timings['rank_alignment']:.3f}s"
        )
        process_group_started = time.perf_counter()
        WorkflowTaskLogger.get_current().log(
            f"easyscale process-group-init start generation={plan.generation} "
            f"rank={plan.physical_rank} world={plan.physical_world_size} "
            f"backend={plan.backend}"
        )
        dist.init_process_group(
            backend=plan.backend,
            store=store,
            rank=plan.physical_rank,
            world_size=plan.physical_world_size,
            timeout=timedelta(seconds=timeout),
        )
        timings["process_group_init"] = time.perf_counter() - process_group_started
        timings["rendezvous"] = time.perf_counter() - rendezvous_started
        WorkflowTaskLogger.get_current().log(
            f"easyscale process-group-init done generation={plan.generation} "
            f"rank={plan.physical_rank} elapsed={timings['process_group_init']:.3f}s "
            f"rendezvous_total={timings['rendezvous']:.3f}s"
        )
        started = True
        yield rendezvous_host, rendezvous_port, timings
    finally:
        try:
            if started and dist.is_initialized():
                teardown_started = time.perf_counter()
                WorkflowTaskLogger.get_current().log(
                    f"easyscale process-group-teardown start "
                    f"generation={plan.generation} rank={plan.physical_rank}"
                )
                dist.destroy_process_group()
                WorkflowTaskLogger.get_current().log(
                    f"easyscale process-group-teardown done "
                    f"generation={plan.generation} rank={plan.physical_rank} "
                    f"elapsed={time.perf_counter() - teardown_started:.3f}s"
                )
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
    boundary_decision: Callable[[int, float | None, bool], GenerationAction] | None,
    boundary_retry: Callable[[], bool] | None,
    runtime_plugin=None,
) -> GenerationResult:
    body_started = time.perf_counter()
    seed = int(getattr(training_config, "seed", 0))
    virtual_world_size = len(state.contexts)
    WorkflowTaskLogger.get_current().log(
        f"easyscale generation-body start generation={plan.generation} "
        f"rank={plan.physical_rank} world={plan.physical_world_size} "
        f"ests={plan.assigned_ests} device={device} "
        f"checkpoint={state.checkpoint_path or 'none'}"
    )
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
    )
    virtual_loader: VirtualDataLoader | None = None
    if runtime_plugin is not None and train_loader.persistent_workers:
        virtual_loader = runtime_plugin.take_dataloader(plan.job_id)
    if virtual_loader is None:
        virtual_loader = build_virtual_dataloader(
            train_loader,
            virtual_world_size,
            seed,
            assigned_ranks=plan.assigned_ests,
            contexts=assigned_contexts,
        )
    else:
        virtual_loader.reconfigure(plan.assigned_ests, assigned_contexts)
        virtual_loader.restore_workers()
    if runtime_plugin is not None and virtual_loader.persistent_workers:
        runtime_plugin.retain_dataloader(plan.job_id, virtual_loader)
    WorkflowTaskLogger.get_current().log(
        f"easyscale virtual-dataloader ready generation={plan.generation} "
        f"rank={plan.physical_rank} ests={plan.assigned_ests} "
        f"elapsed={time.perf_counter() - body_started:.3f}s"
    )
    control_callback = EasyScaleControlCallback(
        strategy,
        boundary_decision,
        training_config.easyscale_scheduler.control_interval_steps,
        boundary_retry,
    )
    checkpoint_callback = EasyScaleCheckpointCallback(
        strategy,
        plan.checkpoint_path,
        control_callback,
        rank_zero_state=state.rank_zero_state,
        persistent_loader=virtual_loader if runtime_plugin is not None else None,
    )
    callbacks: list[L.Callback] = copy.deepcopy(list(external_callbacks))
    patience = getattr(training_config, "patience", None)
    if patience is not None:
        if val_loader is None:
            raise ValueError("EasyScale patience requires a validation loader.")
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))
    callbacks.extend((control_callback, checkpoint_callback))

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
        reload_dataloaders_every_n_epochs=1,
        precision="32-true",
    )
    WorkflowTaskLogger.get_current().log(
        f"easyscale trainer ready generation={plan.generation} "
        f"rank={plan.physical_rank} elapsed={time.perf_counter() - body_started:.3f}s"
    )
    if state.checkpoint_path is not None:
        checkpoint = Path(state.checkpoint_path)
        checkpoint_size = checkpoint.stat().st_size
        WorkflowTaskLogger.get_current().log(
            f"easyscale checkpoint preflight generation={plan.generation} "
            f"rank={plan.physical_rank} path={checkpoint} bytes={checkpoint_size}"
        )
    fit_started = time.perf_counter()
    WorkflowTaskLogger.get_current().log(
        f"easyscale trainer-fit start generation={plan.generation} "
        f"rank={plan.physical_rank} checkpoint={state.checkpoint_path or 'none'}"
    )
    trainer.fit(
        model,
        train_dataloaders=virtual_loader,
        val_dataloaders=val_loader,
        ckpt_path=state.checkpoint_path,
    )
    fit_seconds = time.perf_counter() - fit_started
    WorkflowTaskLogger.get_current().log(
        f"easyscale trainer-fit done generation={plan.generation} "
        f"rank={plan.physical_rank} elapsed={fit_seconds:.3f}s "
        f"checkpoint_saved={checkpoint_callback.saved}"
    )
    if not checkpoint_callback.saved:
        virtual_loader.close_workers()
        if runtime_plugin is not None:
            runtime_plugin.close_job(plan.job_id)
        raise RuntimeError("EasyScale generation stopped without a state checkpoint.")
    if control_callback.action == GenerationAction.RECONFIGURE:
        virtual_loader.park_workers()
    elif runtime_plugin is not None:
        runtime_plugin.close_job(plan.job_id)

    contexts = strategy.all_contexts
    if contexts is None:
        raise RuntimeError("EasyScale generation did not gather EST contexts.")
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

    profile_steps = strategy.profile_steps
    profile_means = {
        f"profile_mean_{name}": total / profile_steps
        for name, total in strategy.profile_totals.items()
        if profile_steps
        and name not in {"est_count", "parameter_count"}
    }
    control_events = control_callback.control_profile_events
    if control_events:
        profile_means.update(
            {
                f"profile_mean_{name}": total / control_events
                for name, total in control_callback.control_profile_totals.items()
            }
        )

    return GenerationResult(
        job_id=plan.job_id,
        generation=plan.generation,
        physical_rank=plan.physical_rank,
        physical_world_size=plan.physical_world_size,
        assigned_ests=plan.assigned_ests,
        checkpoint_path=plan.checkpoint_path,
        target_epoch=control_callback.completed_epoch,
        throughput=control_callback.throughput,
        contexts=contexts,
        should_stop=bool(trainer.should_stop),
        action=control_callback.action,
        rendezvous_host=rendezvous_host,
        rendezvous_port=rendezvous_port,
        rank_zero_state=rank_zero_state,
        timings={
            "worker_setup": fit_started - body_started,
            "trainer_fit": fit_seconds,
            "boundary_control": control_callback.control_seconds,
            "checkpoint": checkpoint_callback.save_seconds,
            "useful_fit": max(
                fit_seconds
                - control_callback.control_seconds
                - checkpoint_callback.save_seconds,
                0.0,
            ),
            "generation_body": time.perf_counter() - body_started,
            "profile_steps": float(profile_steps),
            **profile_means,
        },
    )


def close_easyscale_dataloaders(job_id: str, dask_worker=None) -> None:
    if dask_worker is None:
        return
    
    runtime_plugin = getattr(dask_worker, "plugins", {}).get(EASYSCALE_RUNTIME_PLUGIN_NAME)
    if runtime_plugin is not None:
        runtime_plugin.close_job(job_id)


def run_easyscale_generation(
    plan: GenerationPlan,
    state: GenerationState,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    training_config,
    model_tag: str | None = None,
    external_callbacks: Sequence[L.Callback] = (),
    boundary_decision: Callable[[int, float | None, bool], GenerationAction] | None = None,
    boundary_retry: Callable[[], bool] | None = None,
    rendezvous_store: dist.TCPStore | None = None,
) -> GenerationResult:
    if isinstance(model, bytes):
        model = cloudpickle.loads(model)
        train_loader = cloudpickle.loads(train_loader)
        val_loader = cloudpickle.loads(val_loader)
    task_started = time.perf_counter()
    wall_started = time.time()
    client, worker = connect_dask()
    runtime_plugin = getattr(worker, "plugins", {}).get(EASYSCALE_RUNTIME_PLUGIN_NAME)
    if worker is not None:
        WorkflowTaskLogger.get_current().log(
            f"EasyScale generation {plan.generation} rank {plan.physical_rank} "
            f"world {plan.physical_world_size} ests={plan.assigned_ests} "
            f"running on worker {worker.address}"
        )
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
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with framework_state_preserved(device):
        rendezvous_started = time.perf_counter()
        with process_group(plan, client, worker, timeout, rendezvous_store) as (
            rendezvous_host,
            rendezvous_port,
            rendezvous_timings,
        ):
            rendezvous_seconds = time.perf_counter() - rendezvous_started
            WorkflowTaskLogger.get_current().log(
                f"easyscale rendezvous complete generation={plan.generation} "
                f"rank={plan.physical_rank} world={plan.physical_world_size} "
                f"endpoint={rendezvous_timings.get('endpoint', 0.0):.3f}s "
                f"store={rendezvous_timings.get('store', 0.0):.3f}s "
                f"rank_alignment={rendezvous_timings.get('rank_alignment', 0.0):.3f}s "
                f"process_group_init={rendezvous_timings.get('process_group_init', 0.0):.3f}s "
                f"total={rendezvous_seconds:.3f}s"
            )
            try:
                result = _run_generation_body(
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
                    boundary_decision,
                    boundary_retry,
                    runtime_plugin,
                )
            except Exception:
                if runtime_plugin is not None:
                    runtime_plugin.close_job(plan.job_id)
                raise
        process_group_teardown = (
            time.perf_counter() - rendezvous_started - rendezvous_seconds
            - result.timings["generation_body"]
        )
    return dataclasses.replace(
        result,
        timings={
            **result.timings,
            **rendezvous_timings,
            "process_group_teardown": max(process_group_teardown, 0.0),
            "task_total": time.perf_counter() - task_started,
            "wall_started": wall_started,
            "wall_finished": time.time(),
        },
    )
