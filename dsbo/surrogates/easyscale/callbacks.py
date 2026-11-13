from __future__ import annotations

import torch
import math
import pickle
import time
import lightning as L
from collections.abc import Callable
from typing import Any
from pathlib import Path

from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.random import VirtualRankContext, _capture_random_state, _restore_random_state
from dsbo.surrogates.easyscale.generation import GenerationAction
from dsbo.surrogates.easyscale.virtual_dataset import VirtualBatch, VirtualDataLoader


def measure_performance(
    strategy: EasyScaleStrategy,
    completed_steps: int,
    elapsed: float,
    compute_steps: int,
    compute_seconds: float,
) -> tuple[float, tuple[float | None, ...]]:
    samples = strategy.all_gather(
        torch.tensor(
            [completed_steps, elapsed, compute_steps, compute_seconds],
            dtype=torch.float64,
            device=strategy.root_device,
        )
    ).cpu().tolist()
    throughput = sum(sample[0] for sample in samples) / max(max(sample[1] for sample in samples), torch.finfo(torch.float64).eps)
    capabilities = []
    for _, _, steps, seconds in samples:
        rate = steps / seconds if steps > 0 and seconds > 0 else 0.0
        capabilities.append(rate if math.isfinite(rate) and rate > 0 else None)
    return throughput, tuple(capabilities)


def synchronize_action(
    strategy: EasyScaleStrategy,
    decide: Callable[[int, float | None, bool], GenerationAction] | None,
    should_retry: Callable[[], bool] | None,
    completed_epoch: int,
    throughput: float | None,
    finished: bool,
    capabilities: tuple[float | None, ...] | None = None,
    capability_report: Callable[[tuple[float | None, ...]], None] | None = None,
) -> tuple[GenerationAction, bool, Exception | None]:
    failure = None
    action = GenerationAction.CONTINUE
    retry = False
    if strategy.is_global_zero:
        try:
            if capabilities is not None and capability_report is not None:
                capability_report(capabilities)
            if decide is not None:
                action = decide(completed_epoch, throughput, finished)
            elif finished:
                action = GenerationAction.STOP
            retry = should_retry() if should_retry is not None else False
        except Exception as exc:
            failure = exc
            action = GenerationAction.FAIL

    #encodes the GenerationAction enum as a GPU/CPU scalar tensor so all
    # ranks can receive the value through the process group
    encoded = torch.tensor([int(action), int(retry)], dtype=torch.int64, device=strategy.root_device)
    strategy.broadcast(encoded, src=0)
    return GenerationAction(int(encoded[0].item())), bool(encoded[1].item()), failure


class EasyScaleControlCallback(L.Callback):
    """Synchronize controller decisions at completed global-step boundaries."""

    def __init__(
        self,
        strategy: EasyScaleStrategy,
        decide: Callable[[int, float | None, bool], GenerationAction] | None,
        control_interval_steps: int,
        should_retry: Callable[[], bool] | None = None,
        capability_report: Callable[[tuple[float | None, ...]], None] | None = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.decide = decide
        self.should_retry = should_retry
        self.capability_report = capability_report
        self.capabilities: tuple[float | None, ...] = ()
        self.control_interval_steps = control_interval_steps
        self.action = GenerationAction.CONTINUE
        self.throughput = 0.0
        self.completed_epoch = 0
        self.control_seconds = 0.0
        self.retry_next_step = False
        self._last_steps = 0
        self._last_time = 0.0
        self._compute_steps = 0
        self._compute_seconds = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_steps = self.strategy.est_steps
        self._last_time = time.perf_counter()
        self._compute_steps = self.strategy.est_steps
        self._compute_seconds = self.strategy.profile_totals.get("est_compute", 0.0)

    def _synchronize_boundary_decision(
        self,
        trainer,
        completed_epoch: int,
        finished: bool,
        profile: bool = True,
    ) -> None:
        control_started = time.perf_counter()
        self.completed_epoch = completed_epoch
        measured_throughput = None
        capabilities = None
        if profile:
            now = time.perf_counter()
            compute_seconds = self.strategy.profile_totals.get("est_compute", 0.0)
            self.throughput, samples = measure_performance(
                self.strategy,
                self.strategy.est_steps - self._last_steps,
                now - self._last_time,
                self.strategy.est_steps - self._compute_steps,
                compute_seconds - self._compute_seconds,
            )
            previous = self.capabilities or (None,) * len(samples)
            self.capabilities = tuple(old if sample is None else sample if old is None else 0.5 * old + 0.5 * sample for old, sample in zip(previous, samples))
            if any(sample is not None for sample in samples):
                capabilities = self.capabilities
            self._last_steps = self.strategy.est_steps
            self._last_time = now
            self._compute_steps = self.strategy.est_steps
            self._compute_seconds = compute_seconds
            measured_throughput = self.throughput
        self.action, self.retry_next_step, failure = synchronize_action(
            self.strategy,
            self.decide,
            self.should_retry,
            self.completed_epoch,
            measured_throughput,
            finished,
            capabilities,
            self.capability_report,
        )
        if self.action != GenerationAction.CONTINUE:
            trainer.should_stop = True
        self.control_seconds += time.perf_counter() - control_started
        if self.action == GenerationAction.FAIL:
            if failure is not None:
                raise failure
            raise RuntimeError("EasyScale controller failed at a training boundary.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.strategy.profile_steps == 1:
            # The first completed step is warm-up, not a capability sample.
            self._compute_steps = self.strategy.est_steps
            self._compute_seconds = self.strategy.profile_totals.get("est_compute", 0.0)
        elif isinstance(batch, VirtualBatch) and any(len(indices) != trainer.train_dataloader.batch_sampler.batch_size for indices in batch.indices):
            # A short final mini-batch is not representative of full-batch capacity.
            self._compute_steps += len(batch.ranks)
            self._compute_seconds += self.strategy.last_step_profile["est_compute"]
        if self.action != GenerationAction.CONTINUE:
            return
        if batch_idx + 1 >= trainer.num_training_batches:
            return
        profile = self.strategy.profile_steps % self.control_interval_steps == 0
        if not profile and not self.retry_next_step:
            return
        self._synchronize_boundary_decision(
            trainer,
            trainer.current_epoch,
            bool(trainer.should_stop),
            profile=profile,
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.action != GenerationAction.CONTINUE:
            return
        completed_epoch = trainer.current_epoch + 1
        self._synchronize_boundary_decision(
            trainer,
            completed_epoch,
            bool(trainer.should_stop) or completed_epoch >= trainer.max_epochs,
        )


class EasyScaleCheckpointCallback(L.Callback):
    def __init__(
        self,
        strategy: EasyScaleStrategy,
        checkpoint_path: str,
        control: EasyScaleControlCallback,
        rank_zero_state=None,
        persistent_loader: VirtualDataLoader | None = None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.checkpoint_path = checkpoint_path
        self.control = control
        self.rank_zero_state = rank_zero_state
        self.persistent_loader = persistent_loader
        self.saved = False
        self.save_seconds = 0.0
        self._saving_mid_epoch = False

    @property
    def state_key(self) -> str:
        return "dsbo.easyscale.est-contexts"

    def state_dict(self) -> dict[str, Any]:
        contexts = self.strategy.all_contexts or tuple(self.strategy.contexts.values())
        serialized_state = pickle.dumps(
            {
                "contexts": contexts,
                "rank_zero_state": self.rank_zero_state,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return {"serialized_state": torch.tensor(list(serialized_state), dtype=torch.uint8)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        tensor = state_dict.get("serialized_state")        
        serialized_state = pickle.loads(bytes(tensor.cpu().tolist()))
        restored = {context.rank: context for context in serialized_state["contexts"]}
        
        for rank, context in self.strategy.contexts.items():
            source = restored[rank]
            context.random_state = source.random_state
            context.sampler_epoch = source.sampler_epoch
            context.sampler_position = source.sampler_position
        self.rank_zero_state = serialized_state.get("rank_zero_state")

    def on_train_start(self, trainer, pl_module) -> None:
        if self.rank_zero_state is not None:
            _restore_random_state(
                self.rank_zero_state,
                self.strategy.root_device,
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if not self._saving_mid_epoch:
            return
        # The virtual sampler already omits consumed batches, so resume this
        # shortened loader as the start of the still-incomplete epoch.
        epoch_progress = checkpoint["loops"]["fit_loop"]["epoch_progress"]
        for progress in epoch_progress.values():
            for name in progress:
                progress[name] = self.control.completed_epoch

    def save(self, trainer) -> None:
        if self.saved:
            return
        save_started = time.perf_counter()
        if self.strategy._pending is not None:
            raise RuntimeError("cannot checkpoint a partial EasyScale EST batch")
        contexts = self.strategy.gather_contexts()
        boundaries = {
            (context.sampler_epoch, context.sampler_position) for context in contexts
        }
        if len(boundaries) != 1:
            raise RuntimeError("EST contexts are not at one completed global-step boundary")
        self._saving_mid_epoch = contexts[0].sampler_position != 0
        self.rank_zero_state = (
            _capture_random_state(self.strategy.root_device)
            if self.strategy.is_global_zero
            else None
        )
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.checkpoint_path)
        self.strategy.barrier()
        self.saved = True
        self.save_seconds += time.perf_counter() - save_started

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.control.action not in {
            GenerationAction.RECONFIGURE,
            GenerationAction.STOP,
        }:
            return
        self.save(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.control.action not in {
            GenerationAction.RECONFIGURE,
            GenerationAction.STOP,
        }:
            return
        self.save(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        if (
            self.control.action == GenerationAction.RECONFIGURE
            and self.persistent_loader is not None
        ):
            self.persistent_loader.park_workers()
