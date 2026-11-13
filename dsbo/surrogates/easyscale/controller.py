from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import dask
import lightning as L
import torch.distributed as dist
from distributed import Client, get_worker, wait
from distributed.client import FuturesCancelledError
from torch.utils.data import DataLoader

from dsbo.surrogates.easyscale.allocator import Capacity, ClientResourceAllocator, Registration
from dsbo.surrogates.easyscale.generation import (
    GenerationAction,
    GenerationPlan,
    GenerationResult,
    GenerationState,
)
from dsbo.surrogates.easyscale.protocol import ESTPlan, ExecutionLayout, PlanDatabase, assign_virtual_ranks
from dsbo.surrogates.easyscale.random import VirtualRankContext, initial_context
from dsbo.surrogates.easyscale.runner import (
    close_easyscale_dataloaders,
    run_easyscale_generation,
)
from dsbo.utils.io import Workdir
from dsbo.worker.spec import WorkerSpec
from dsbo.logger import WorkflowTaskLogger
from dsbo.storage import EasyScaleStorageBase, get_current_storage
from dsbo.storage.models import EasyScaleGeneration, JobStatus
from dsbo.surrogates.easyscale.profile import profile_details

if TYPE_CHECKING:
    from dsbo.surrogates.surrogate_model import SurrogateModel
    from dsbo.surrogates.trainer import TrainingConfig


def _dist_backend(resources: WorkerSpec) -> str:
    if resources.num_gpus == 0:
        return "gloo"
    if resources.num_gpus != 1:
        raise ValueError("EasyScale requires exactly one GPU per worker.")
    return "nccl"


class WorkerStateMachine:
    def __init__(
        self,
        allocator: ClientResourceAllocator,
        database: PlanDatabase,
        job_id: str,
        virtual_world_size: int,
        resource_type: str,
        max_worker_count: int,
        registration: Registration,
    ) -> None:
        self.allocator = allocator
        self.database = database
        self.job_id = job_id
        self.virtual_world_size = virtual_world_size
        self.resource_type = resource_type
        self.max_worker_count = max_worker_count
        self.allocation_revision = registration.allocation_revision
        self.active_worker_count = self._worker_count(registration.allocation)
        self.approved_worker_count = self.active_worker_count
        self.pending_round: int | None = None
        self.current_plan: ESTPlan | None = None
        self._candidate_worker_count = self.active_worker_count
        self._capacity_ready_since: float | None = None
        self._capacity_changed_at = 0.0
        self._generation_started_at: float | None = float("-inf")

    def select_active_plan(self) -> ESTPlan:
        plan = (
            self.database.initial_plan(self.active_worker_count)
            if self.database.capability is None
            else self.database.select_plan(self.active_worker_count)
        )

        if plan != self.current_plan:
            self.database.adopt_plan(plan)
            self.current_plan = plan

        return plan

    def needs_retry(self) -> bool:
        return (
            self.pending_round is not None
            or self.approved_worker_count != self.active_worker_count
        )

    def update_execution_layout(self, layout: ExecutionLayout) -> None:
        self.database.update_execution_layout(layout)
        self.current_plan = self.database.current_plan

    def update_capabilities(self, capabilities: Sequence[float | None]) -> None:
        self.database.update_capabilities(capabilities)
        self.current_plan = self.database.current_plan

    def decide(
        self,
        epoch: int,
        throughput: float | None,
        should_stop: bool,
    ) -> GenerationAction:
        now = time.monotonic()
        # Start the cooldown at the first completed boundary, excluding generation setup.
        if self._generation_started_at is None:
            self._generation_started_at = now
        self._poll_allocation()

        if throughput is not None:
            self.database.worker_hardware = self.allocator.capacity(self.job_id).worker_hardware

        if throughput is not None:
            if self.database.current_plan is None:
                self.select_active_plan()
            fallback = self.database.fallback_if_slowdown(throughput)
            self.current_plan = self.database.current_plan
            if fallback is not None:
                decision = self.allocator.trigger_fallback(
                    self.job_id,
                    {self.resource_type: fallback.resource_count},
                    self.allocation_revision,
                )
                self._apply_approved(decision.allocation, decision.allocation_revision)
                self.active_worker_count = self.approved_worker_count
                self.pending_round = None
                self._generation_started_at = None
                self._capacity_ready_since = None
                return GenerationAction.RECONFIGURE

        if should_stop:
            self._release_final_tail()
            return GenerationAction.STOP

        if throughput is not None and self.pending_round is None and not self.needs_retry():
            plan = self.database.current_plan
            proposals = () if self.database.capability is None else self.database.propose(plan, self.max_worker_count)
            submission = self.allocator.submit_proposals(self.job_id, proposals)
            if submission.pending:
                self.pending_round = submission.round_id
            elif submission.decision is not None:
                self._apply_approved(
                    submission.decision.allocation,
                    submission.decision.allocation_revision,
                )

        if self.approved_worker_count != self.active_worker_count:
            capacity = self.allocator.capacity(self.job_id)
            self.database.worker_hardware = capacity.worker_hardware
            if self._apply_approved(capacity.approved_allocation, capacity.allocation_revision):
                return self._adopt_ready_capacity_incrementally(capacity, now)
        return GenerationAction.CONTINUE

    def _adopt_ready_capacity_incrementally(self, capacity: Capacity, now: float) -> GenerationAction:
        worker_count = self._select_scale_out_worker_count(capacity)
        if worker_count <= self.active_worker_count:
            self._capacity_ready_since = None
            return GenerationAction.CONTINUE
        if not self._partial_scale_out_improves_throughput(worker_count):
            self._capacity_ready_since = None
            return GenerationAction.CONTINUE
        self._track_scale_out_candidate(worker_count, now)
        if not self._scale_out_candidate_can_be_adopted(worker_count, now):
            return GenerationAction.CONTINUE
        self.active_worker_count = worker_count
        self.select_active_plan()
        self._generation_started_at = None
        self._capacity_ready_since = None
        return GenerationAction.RECONFIGURE

    def _select_scale_out_worker_count(self, capacity: Capacity) -> int:
        ready_live = self._worker_count(capacity.ready_live)
        approved_total = self._worker_count(capacity.approved_total)
        # Preserve other jobs' approvals while adopting ready capacity incrementally.
        reserved_by_others = approved_total - self.approved_worker_count
        available = ready_live - reserved_by_others
        if capacity.ready:
            return self.approved_worker_count
        target = max(self.active_worker_count, available)
        return min(self.approved_worker_count, target)

    def _partial_scale_out_improves_throughput(self, worker_count: int) -> bool:
        if worker_count >= self.approved_worker_count:
            return True
        if self.database.capability is None:
            return True
        candidate = self.database.select_plan(worker_count)
        active = self.database.current_plan
        return candidate.estimated_throughput > active.estimated_throughput

    def _track_scale_out_candidate(self, worker_count: int, now: float) -> None:
        if self._capacity_ready_since is None:
            self._capacity_ready_since = now
            self._capacity_changed_at = now
        if worker_count != self._candidate_worker_count:
            self._candidate_worker_count = worker_count
            self._capacity_changed_at = now

    def _scale_out_candidate_can_be_adopted(self, worker_count: int, now: float) -> bool:
        config = self.database.config
        settled = (
            worker_count == self.approved_worker_count
            or now - self._capacity_changed_at >= config.scale_out_settle_seconds
            or now - self._capacity_ready_since >= config.scale_out_max_wait_seconds
        )
        cooled_down = now - self._generation_started_at >= config.min_generation_seconds
        return settled and cooled_down

    def _worker_count(self, allocation: Mapping[str, int]) -> int:
        return int(allocation.get(self.resource_type, 0))

    def _apply_approved(self, allocation: Mapping[str, int], revision: int) -> bool:
        worker_count = self._worker_count(allocation)
        if worker_count > self.virtual_world_size:
            raise RuntimeError("granted physical allocation exceeds virtual world size.")
        if revision < self.allocation_revision:
            return False
        if revision == self.allocation_revision:
            if worker_count != self.approved_worker_count:
                raise RuntimeError("allocation changed without advancing its revision.")
            return True
        self.allocation_revision = revision
        self.approved_worker_count = worker_count
        return True

    def _poll_allocation(self) -> None:
        if self.pending_round is None:
            return
        decision = self.allocator.poll_allocation(self.job_id, self.pending_round)
        if decision is not None:
            self.pending_round = None
            self._apply_approved(decision.allocation, decision.allocation_revision)

    def _release_final_tail(self) -> None:
        final_count = min(self.approved_worker_count, self.active_worker_count)
        if self.pending_round is not None or self.approved_worker_count != final_count:
            decision = self.allocator.release_allocation(
                self.job_id,
                {self.resource_type: final_count},
                self.allocation_revision,
                "final_tail",
            )
            self._apply_approved(decision.allocation, decision.allocation_revision)
        self.pending_round = None


@dataclass
class GenerationRunner:
    client: Client
    allocator: ClientResourceAllocator
    controller_worker: Any
    resource_type: str
    backend: str
    model: SurrogateModel
    train_loader: DataLoader
    val_loader: DataLoader | None
    config: TrainingConfig
    model_tag: str
    callbacks: Sequence[L.Callback]

    def run(
        self,
        job_id: str,
        generation: int,
        target_epoch: int,
        checkpoint_path: str,
        state: GenerationState,
        assignments: Mapping[str, Sequence[int]],
        boundary_decision,
        boundary_retry,
        execution_ready: Callable[[ExecutionLayout], None] | None = None,
        capability_report: Callable[[tuple[float | None, ...]], None] | None = None,
    ) -> tuple[list[GenerationResult], dict[str, Any]]:
        slots = sorted(assignments)
        store, host, port = self._rendezvous(len(slots))
        plans = [
            GenerationPlan(
                job_id=job_id,
                generation=generation,
                physical_rank=rank,
                physical_world_size=len(slots),
                assigned_ests=assignments[slot],
                checkpoint_path=checkpoint_path,
                target_epoch=target_epoch,
                backend=self.backend,
                rendezvous_host=host,
                rendezvous_port=port,
            )
            for rank, slot in enumerate(slots)
        ]
        submission_started = time.perf_counter()
        scatter_seconds = 0.0
        futures: list[Any] = []

        try:
            WorkflowTaskLogger.get_current().log(
                f"easyscale generation dispatch generation={generation} "
                f"world={len(plans)} "
                f"assignments={assignments} checkpoint={checkpoint_path or 'none'}"
            )
            nonzero_rank_plans = plans[1:]
            if nonzero_rank_plans:
                scatter_started = time.perf_counter()
                model, train_loader, val_loader = self.client.scatter(
                    [
                        cloudpickle.dumps(self.model),
                        cloudpickle.dumps(self.train_loader),
                        cloudpickle.dumps(self.val_loader),
                    ],
                    broadcast=True,
                    hash=False,
                )
                scatter_seconds = time.perf_counter() - scatter_started
                submitted_wall = time.time()
                with dask.annotate(dsbo_easyscale_sibling=True):
                    futures.extend(
                        self.client.map(
                            run_easyscale_generation,
                            nonzero_rank_plans,
                            state=state,
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            training_config=self.config,
                            model_tag=self.model_tag,
                            external_callbacks=self.callbacks,
                            key=[
                                f"dsbo-easyscale-{job_id}-generation-{generation}"
                                f"-rank-{plan.physical_rank}"
                                for plan in nonzero_rank_plans
                            ],
                            resources={self.resource_type: 1},
                            retries=0,
                            pure=False,
                        )
                    )
                task_logger = WorkflowTaskLogger.get_current()
                for future, plan in zip(futures, nonzero_rank_plans):
                    if not hasattr(future, "add_done_callback"):
                        continue

                    def log_rank_failure(completed, rank=plan.physical_rank):
                        try:
                            failure = completed.exception()
                        except Exception as exc:
                            failure = exc
                        if failure is not None:
                            task_logger.log(
                                f"EasyScale generation {generation} rank {rank} failed: "
                                f"{failure!r}",
                                logging.ERROR,
                            )

                    future.add_done_callback(log_rank_failure)
            else:
                submitted_wall = time.time()
            submission_seconds = time.perf_counter() - submission_started
            WorkflowTaskLogger.get_current().log(
                f"easyscale generation siblings-submitted generation={generation} "
                f"count={len(nonzero_rank_plans)} elapsed={submission_seconds:.3f}s"
            )
            results = [
                run_easyscale_generation(
                    plans[0],
                    state,
                    self.model,
                    self.train_loader,
                    self.val_loader,
                    self.config,
                    self.model_tag,
                    self.callbacks,
                    boundary_decision,
                    boundary_retry,
                    store,
                    execution_ready,
                    capability_report,
                )
            ]
            gather_started = time.perf_counter()
            if futures:
                results.extend(self.client.gather(futures))
            gather_seconds = time.perf_counter() - gather_started if futures else 0.0
            for result in sorted(results, key=lambda item: item.physical_rank):
                timings = result.timings
                steps = int(timings.get("profile_steps", 0.0))
                if not steps:
                    continue
                WorkflowTaskLogger.get_current().log(
                    f"easyscale generation profile generation={generation} "
                    f"rank={result.physical_rank} steps={steps} "
                    f"ests={result.assigned_ests} "
                    f"gradient_reduction=all_reduce "
                    f"gradient_bytes={result.gradient_bytes} "
                    f"{profile_details(timings)}"
                )
        except Exception:
            try:
                if futures:
                    self.client.cancel(futures)
                    with suppress(FuturesCancelledError):
                        wait(futures)
            except Exception as cleanup_error:
                WorkflowTaskLogger.get_current().log(
                    f"easyscale generation cleanup failed "
                    f"generation={generation} error={cleanup_error!r}",
                    logging.ERROR,
                )
            finally:
                self.allocator.fail_generation(job_id, generation)
            raise

        return results, {
            "world_size": len(slots),
            "submitted_wall": submitted_wall,
            "scatter": scatter_seconds,
            "submission": submission_seconds,
            "gather": gather_seconds,
            "ranks": [
                {
                    "rank": result.physical_rank,
                    "assigned_ests": result.assigned_ests,
                    "worker": slots[result.physical_rank],
                    "worker_name": result.worker,
                    "worker_address": result.worker_address,
                    "device": result.device,
                    "hardware": result.hardware,
                    "capability": result.capability,
                    "gradient_mb": None if result.gradient_bytes is None else result.gradient_bytes / 1e6,
                    "performance": result.performance,
                    "queue": 0.0 if result.physical_rank == 0 else max(
                        result.timings.get("wall_started", submitted_wall)
                        - submitted_wall,
                        0.0,
                    ),
                    **result.timings,
                }
                for result in sorted(results, key=lambda item: item.physical_rank)
            ],
        }

    def _rendezvous(
        self,
        world_size: int,
    ) -> tuple[dist.TCPStore | None, str | None, int | None]:
        host = getattr(self.controller_worker, "host", None)
        if world_size == 1 or host is None:
            return None, None, None
        store = dist.TCPStore(
            host,
            0,
            world_size,
            True,
            timedelta(seconds=self.config.easyscale_scheduler.allocation_timeout),
            wait_for_workers=False,
        )
        return store, host, store.port


class EasyScaleController:
    """Coordinate one worker-local, multi-generation EasyScale fit."""
    def __init__(
        self,
        client: Client,
        allocator: ClientResourceAllocator | None = None,
    ) -> None:
        self.client = client
        self.allocator = (
            ClientResourceAllocator(client) if allocator is None else allocator
        )
        self.generation_profiles: list[dict[str, Any]] = []

    def fit(
        self,
        model: SurrogateModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        training_config: TrainingConfig,
        resources: WorkerSpec,
        model_tag: str,
        resume_state: GenerationState | None = None,
        callbacks: Sequence[L.Callback] = (),
    ) -> None:
        controller_worker = get_worker()
        job_id = f"{model_tag or 'surrogate'}-{uuid.uuid4().hex}"
        resource_type = resources.resource_key
        owned_checkpoints: set[str] = set()
        succeeded = False
        registered = False
        easyscale_storage = get_current_storage().easyscale

        try:
            registration = self.allocator.register_job(
                job_id,
                priority=int(getattr(training_config, "priority", 0)),
                initial_allocation={resource_type: resources.min_worker},
            )
            registered = True
            maximum = resources.max_worker
            allocation = WorkerStateMachine(
                self.allocator,
                PlanDatabase(
                    training_config.virtual_world_size,
                    resource_type,
                    training_config.easyscale_scheduler,
                ),
                job_id,
                training_config.virtual_world_size,
                resource_type,
                min(
                    registration.provision_limits.get(resource_type, maximum),
                    maximum,
                ),
                registration,
            )
            runner = GenerationRunner(
                self.client,
                self.allocator,
                controller_worker,
                resource_type,
                _dist_backend(resources),
                model,
                train_loader,
                val_loader,
                training_config,
                model_tag,
                callbacks,
            )

            generation = registration.generation
            checkpoint_path = None if resume_state is None else resume_state.checkpoint_path
            completed_epoch = 0 if resume_state is None else resume_state.completed_epoch
            contexts: tuple[VirtualRankContext, ...] = (
                tuple(
                    initial_context(rank, int(training_config.seed))
                    for rank in range(training_config.virtual_world_size)
                )
                if resume_state is None
                else resume_state.contexts
            )
            result: GenerationResult | None = None

            while completed_epoch < training_config.max_epochs:
                workers = tuple(
                    f"{resource_type}-{index}"
                    for index in range(allocation.active_worker_count)
                )
                assignments = assign_virtual_ranks(
                    training_config.virtual_world_size,
                    workers,
                    allocation.select_active_plan(),
                )
                next_checkpoint = str(
                    Path(Workdir.get_current_dir())
                    / ".easyscale"
                    / f"{job_id}-generation-{generation}"
                    f"-epoch-{training_config.max_epochs}.ckpt"
                )
                state = GenerationState(
                    checkpoint_path=checkpoint_path,
                    completed_epoch=completed_epoch,
                    contexts=contexts,
                    rank_zero_state=(
                        resume_state.rank_zero_state
                        if resume_state is not None
                        else None
                    ),
                )
                result = self._run_generation_with_telemetry(
                    easyscale_storage, runner, allocation, resources,
                    job_id, generation, next_checkpoint, state, assignments,
                )
                if checkpoint_path in owned_checkpoints and os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
                    owned_checkpoints.discard(checkpoint_path)
                owned_checkpoints.add(next_checkpoint)
                checkpoint_path = next_checkpoint
                contexts = result.contexts
                completed_epoch = result.target_epoch
                resume_state = GenerationState(
                    checkpoint_path=checkpoint_path,
                    completed_epoch=completed_epoch,
                    contexts=contexts,
                    rank_zero_state={
                        key: value
                        for key, value in (result.rank_zero_state or {}).items()
                        if key != "model_state"
                    },
                )

                generation += 1
                if (completed_epoch >= training_config.max_epochs
                    or result.action == GenerationAction.STOP):
                    break

            if result is None:
                raise RuntimeError("EasyScale fit completed without a generation result.")
            model.load_state_dict(result.model_state, strict=True)
            succeeded = True
        finally:
            self._cleanup(job_id, owned_checkpoints, succeeded, registered)

    def _open_generation(
        self,
        easyscale_storage: EasyScaleStorageBase,
        runner: GenerationRunner,
        allocation: WorkerStateMachine,
        resources: WorkerSpec,
        job_id: str,
        generation: int,
        assignments: Mapping[str, Sequence[int]],
        completed_epoch: int,
    ) -> EasyScaleGeneration:
        record = EasyScaleGeneration(
            exec_id=str(Workdir.get_current_dir()), job_id=job_id, model_tag=runner.model_tag,
            generation=generation, world_size=len(assignments),
            virtual_world_size=runner.config.virtual_world_size,
            backend=runner.backend, gradient_reduction="all_reduce",
            started=time.time(), completed_epoch=completed_epoch,
            target_epoch=runner.config.max_epochs,
            config={
                "training": asdict(runner.config),
                "resources": {key: getattr(resources, key, None) for key in (
                    "cores", "processes", "memory", "num_gpus",
                    "min_worker", "max_worker", "walltime", "partition")},
                "allocation_revision": allocation.allocation_revision,
                "plan": asdict(allocation.current_plan),
            },
            ranks=[{"rank": rank, "worker": slot, "assigned_ests": list(assignments[slot])}
                   for rank, slot in enumerate(sorted(assignments))],
        )
        easyscale_storage.open_generation(record)
        return record

    def _run_generation(
        self,
        runner: GenerationRunner,
        allocation: WorkerStateMachine,
        job_id: str,
        generation: int,
        checkpoint_path: str,
        state: GenerationState,
        assignments: Mapping[str, Sequence[int]],
    ) -> tuple[GenerationResult, dict[str, Any]]:
        results, profile = runner.run(
            job_id, generation, runner.config.max_epochs, checkpoint_path,
            state, assignments, allocation.decide, allocation.needs_retry,
            allocation.update_execution_layout, allocation.update_capabilities,
        )
        self.generation_profiles.append({"generation": generation, **profile})
        final_result = next(result for result in results if result.physical_rank == 0)
        self.allocator.complete_generation(job_id, generation, final_result.throughput)
        return final_result, profile

    def _close_generation(
        self,
        record: EasyScaleGeneration,
        easyscale_storage: EasyScaleStorageBase,
        profile: dict[str, Any] | None,
        generation_error: BaseException | None,
        final_result: GenerationResult | None,
    ) -> None:
        record.finished = time.time()
        if profile is not None:
            record.sqlmodel_update({key: profile[key] for key in ("scatter", "submission", "gather", "ranks")})
        if generation_error is None:
            record.sqlmodel_update({"completed_epoch": final_result.target_epoch, 
                                    "throughput": final_result.throughput, 
                                    "action": final_result.action.name, 
                                    "status": JobStatus.SUCCESS})
        else:
            record.sqlmodel_update({"status": JobStatus.FAILED, 
                                    "error": f"{type(generation_error).__name__}: {generation_error}", 
                                    "action": GenerationAction.FAIL.name})
        easyscale_storage.close_generation(record)

    def _run_generation_with_telemetry(
        self,
        easyscale_storage: EasyScaleStorageBase | None,
        runner: GenerationRunner,
        allocation: WorkerStateMachine,
        resources: WorkerSpec,
        job_id: str,
        generation: int,
        checkpoint_path: str,
        state: GenerationState,
        assignments: Mapping[str, Sequence[int]],
    ) -> GenerationResult:
        record = None if easyscale_storage is None else self._open_generation(
            easyscale_storage, runner, allocation, resources,
            job_id, generation, assignments, state.completed_epoch,
        )
        profile = result = generation_error = None
        try:
            result, profile = self._run_generation(
                runner, allocation, job_id, generation,
                checkpoint_path, state, assignments,
            )
        except BaseException as exc:
            generation_error = exc
            raise
        finally:
            if record is not None:
                self._close_generation(
                    record, easyscale_storage, 
                    profile, generation_error, 
                    result
                )
        return result

    def _cleanup(
        self,
        job_id: str,
        owned_checkpoints: set[str],
        succeeded: bool,
        registered: bool,
    ) -> None:
        cleanup_failure = None
        if registered and hasattr(self.client, "run"):
            try:
                self.client.run(close_easyscale_dataloaders, job_id)
            except Exception as exc:
                cleanup_failure = exc
        try:
            if registered:
                self.allocator.close_job(job_id)
        except Exception:
            if succeeded:
                raise
        finally:
            if succeeded:
                for path in owned_checkpoints:
                    if os.path.exists(path):
                        os.unlink(path)
        if cleanup_failure is not None:
            WorkflowTaskLogger.get_current().log(
                f"EasyScale loader cleanup failed: {cleanup_failure}",
                logging.ERROR,
            )

