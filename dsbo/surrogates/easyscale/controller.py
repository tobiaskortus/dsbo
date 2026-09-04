from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import torch.distributed as dist
from distributed import Client, get_worker, wait
from torch.utils.data import DataLoader

from dsbo.surrogates.easyscale.allocator import ClientResourceAllocator, Registration
from dsbo.surrogates.easyscale.generation import (
    GenerationAction,
    GenerationPlan,
    GenerationResult,
    GenerationState,
)
from dsbo.surrogates.easyscale.protocol import ESTPlan, PlanDatabase, assign_virtual_ranks
from dsbo.surrogates.easyscale.random import VirtualRankContext, initial_context
from dsbo.surrogates.easyscale.runner import run_easyscale_generation
from dsbo.logger import WorkflowTaskLogger
from dsbo.utils.io import Workdir
from dsbo.worker.spec import WorkerSpec

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

    def decide(self, epoch: int, throughput: float, should_stop: bool) -> GenerationAction:
        self._poll_allocation()

        if self.database.capability is None:
            self.database.set_capability(throughput / self.active_worker_count)
            self.current_plan = self.database.select_plan(self.active_worker_count)
            self.database.current_plan = self.current_plan
        else:
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
                return GenerationAction.RECONFIGURE

        if should_stop:
            self._release_final_tail()
            return GenerationAction.STOP

        if self.approved_worker_count != self.active_worker_count:
            capacity = self.allocator.capacity(self.job_id)
            if capacity.allocation_revision != self.allocation_revision:
                self._apply_approved(
                    capacity.approved_allocation,
                    capacity.allocation_revision,
                )
            if capacity.ready and self.approved_worker_count != self.active_worker_count:
                self.active_worker_count = self.approved_worker_count
                self.select_active_plan()
                return GenerationAction.RECONFIGURE
            return GenerationAction.CONTINUE

        if self.pending_round is None:
            plan = self.database.select_plan(self.approved_worker_count)
            proposals = self.database.propose(plan, self.max_worker_count)
            self.pending_round = self.allocator.submit_proposals(self.job_id, proposals)
        return GenerationAction.CONTINUE

    def _worker_count(self, allocation: Mapping[str, int]) -> int:
        return int(allocation.get(self.resource_type, 0))

    def _apply_approved(self, allocation: Mapping[str, int], revision: int) -> None:
        self.allocation_revision = revision
        self.approved_worker_count = self._worker_count(allocation)
        if self.approved_worker_count > self.virtual_world_size:
            raise RuntimeError("granted physical allocation exceeds virtual world size.")

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
        submitted_wall = time.time()
        submission_started = time.perf_counter()
        futures: list[Any] = []

        try:
            WorkflowTaskLogger.get_current().log(
                f"easyscale generation dispatch generation={generation} "
                f"world={len(plans)} "
                f"assignments={assignments} checkpoint={checkpoint_path or 'none'}"
            )
            nonzero_rank_plans = plans[1:]
            if nonzero_rank_plans:
                count = len(nonzero_rank_plans)
                futures.extend(
                    self.client.map(
                        run_easyscale_generation,
                        nonzero_rank_plans,
                        [state] * count,
                        [self.model] * count,
                        [self.train_loader] * count,
                        [self.val_loader] * count,
                        [self.config] * count,
                        [self.model_tag] * count,
                        [self.callbacks] * count,
                        [None] * count,
                        [None] * count,
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
                            if exc is not None:
                                task_logger.log(
                                    f"EasyScale generation {generation} rank {rank} failed: "
                                    f"{failure!r}",
                                    logging.ERROR,
                                )

                    future.add_done_callback(log_rank_failure)
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
                    store,
                )
            ]
            gather_started = time.perf_counter()
            if futures:
                results.extend(self.client.gather(futures))
            profile_fields = (
                ("est_data_transfer", "data_transfer"),
                ("est_compute", "est_compute"),
                ("gradient_metadata_all_reduce", "gradient_metadata_all_reduce"),
                ("gradient_tensor_all_gather", "gradient_tensor_all_gather"),
                ("gradient_aggregation_total", "gradient_aggregation"),
                ("metric_all_reduce", "metric_all_reduce"),
                ("metric_aggregation_total", "metric_aggregation"),
                ("gradient_assignment", "gradient_assignment"),
                ("backward_total", "backward"),
                ("step_total", "step_total"),
                ("throughput_all_gather", "throughput_all_gather"),
                ("action_broadcast", "action_broadcast"),
            )
            for result in sorted(results, key=lambda item: item.physical_rank):
                timings = result.timings
                steps = int(timings.get("profile_steps", 0.0))
                if not steps:
                    continue
                details = " ".join(
                    f"{label}={timings.get(f'profile_mean_{name}', 0.0) * 1000.0:.2f}ms"
                    for name, label in profile_fields
                )
                WorkflowTaskLogger.get_current().log(
                    f"easyscale generation profile generation={generation} "
                    f"rank={result.physical_rank} steps={steps} "
                    f"ests={result.assigned_ests} "
                    f"gradient_mb={timings.get('profile_mean_gradient_elements', 0.0) * 4.0 / 1e6:.2f} "
                    f"{details}"
                )
            gather_seconds = time.perf_counter() - gather_started
        except Exception:
            if futures:
                self.client.cancel(futures)
                wait(futures)
            self.allocator.fail_generation(job_id, generation)
            raise

        return results, {
            "world_size": len(slots),
            "submitted_wall": submitted_wall,
            "submission": submission_seconds,
            "gather": gather_seconds,
            "ranks": [
                {
                    "rank": result.physical_rank,
                    "assigned_ests": result.assigned_ests,
                    "queue": max(
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
            final_result: GenerationResult | None = None

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
                results, profile = runner.run(
                    job_id,
                    generation,
                    training_config.max_epochs,
                    next_checkpoint,
                    state,
                    assignments,
                    allocation.decide,
                )
                self.generation_profiles.append(
                    {"generation": generation, **profile}
                )
                rank_zero = [result for result in results if result.physical_rank == 0]
                if len(rank_zero) != 1 or rank_zero[0].model_state is None:
                    self.allocator.fail_generation(job_id, generation)
                    raise RuntimeError(
                        "EasyScale generation did not return exactly one rank-zero state."
                    )
                if not os.path.exists(next_checkpoint):
                    self.allocator.fail_generation(job_id, generation)
                    raise RuntimeError(
                        "EasyScale generation did not commit its shared checkpoint."
                    )
                final_result = rank_zero[0]
                self.allocator.complete_generation(
                    job_id, generation, final_result.throughput
                )
                if checkpoint_path in owned_checkpoints and os.path.exists(checkpoint_path):
                    os.unlink(checkpoint_path)
                    owned_checkpoints.discard(checkpoint_path)
                owned_checkpoints.add(next_checkpoint)
                checkpoint_path = next_checkpoint
                contexts = final_result.contexts
                completed_epoch = final_result.target_epoch
                resume_state = GenerationState(
                    checkpoint_path=checkpoint_path,
                    completed_epoch=completed_epoch,
                    contexts=contexts,
                    rank_zero_state={
                        key: value
                        for key, value in (final_result.rank_zero_state or {}).items()
                        if key != "model_state"
                    },
                )
                generation += 1
                if (
                    completed_epoch >= training_config.max_epochs
                    or final_result.action == GenerationAction.STOP
                ):
                    break

            if final_result is None:
                raise RuntimeError("EasyScale fit completed without a generation result.")
            model.load_state_dict(final_result.model_state, strict=True)
            succeeded = True
        finally:
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
