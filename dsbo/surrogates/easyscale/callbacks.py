from __future__ import annotations

import torch
import pickle
import time
import lightning as L
from collections.abc import Callable
from typing import Any
from pathlib import Path

from dsbo.surrogates.easyscale.strategy import EasyScaleStrategy
from dsbo.surrogates.easyscale.random import VirtualRankContext, _capture_random_state, _restore_random_state
from dsbo.surrogates.easyscale.generation import GenerationAction


def global_throughput(
    strategy: EasyScaleStrategy,
    completed_steps: int,
    elapsed: float,
    timings: dict[str, float] | None = None,
) -> float:
    if timings is not None and strategy.root_device.type == "cuda":
        torch.cuda.synchronize(strategy.root_device)
    gather_started = time.perf_counter()
    completed_by_rank = strategy.all_gather(
        torch.tensor(
            [completed_steps, elapsed],
            dtype=torch.float64,
            device=strategy.root_device,
        )
    )
    if timings is not None:
        if strategy.root_device.type == "cuda":
            torch.cuda.synchronize(strategy.root_device)
        timings["throughput_all_gather"] = time.perf_counter() - gather_started
    total_steps = completed_by_rank[:, 0].sum()
    max_elapsed = completed_by_rank[:, 1].max()
    return float((total_steps / max_elapsed.clamp_min(torch.finfo(torch.float64).eps)).item())


def synchronize_action(
    strategy: EasyScaleStrategy,
    decide: Callable[[int, float, bool], GenerationAction] | None,
    completed_epoch: int,
    throughput: float,
    finished: bool,
    timings: dict[str, float] | None = None,
) -> tuple[GenerationAction, Exception | None]:
    failure = None
    action = GenerationAction.CONTINUE
    if strategy.is_global_zero:
        try:
            if decide is not None:
                action = decide(completed_epoch, throughput, finished)
            elif finished:
                action = GenerationAction.STOP
        except Exception as exc:
            failure = exc
            action = GenerationAction.FAIL

    #encodes the GenerationAction enum as a GPU/CPU scalar tensor so all 
    # ranks can receive the value through the process group
    encoded = torch.tensor(int(action), dtype=torch.int64, device=strategy.root_device)
    if timings is not None and strategy.root_device.type == "cuda":
        torch.cuda.synchronize(strategy.root_device)
    broadcast_started = time.perf_counter()
    strategy.broadcast(encoded, src=0)
    if timings is not None:
        if strategy.root_device.type == "cuda":
            torch.cuda.synchronize(strategy.root_device)
        timings["action_broadcast"] = time.perf_counter() - broadcast_started
    return GenerationAction(int(encoded.item())), failure


class EasyScaleControlCallback(L.Callback):
    """Synchronize controller decisions at completed epoch boundaries."""

    def __init__(
        self,
        strategy: EasyScaleStrategy,
        decide: Callable[[int, float, bool], GenerationAction] | None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.decide = decide
        self.action = GenerationAction.CONTINUE
        self.throughput = 0.0
        self.completed_epoch = 0
        self.control_seconds = 0.0
        self.control_profile_totals: dict[str, float] = {}
        self.control_profile_events = 0
        self._last_steps = 0
        self._last_time = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_steps = self.strategy.est_steps
        self._last_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        control_started = time.perf_counter()
        control_profile: dict[str, float] = {}
        self.completed_epoch = trainer.current_epoch + 1
        now = time.perf_counter()
        self.throughput = global_throughput(
            self.strategy,
            self.strategy.est_steps - self._last_steps,
            now - self._last_time,
            timings=control_profile,
        )
        self._last_steps = self.strategy.est_steps
        self._last_time = now
        finished = bool(trainer.should_stop) or self.completed_epoch >= trainer.max_epochs
        self.action, failure = synchronize_action(
            self.strategy,
            self.decide,
            self.completed_epoch,
            self.throughput,
            finished,
            timings=control_profile,
        )
        if self.action != GenerationAction.CONTINUE:
            trainer.should_stop = True
        self.control_seconds += time.perf_counter() - control_started
        self.control_profile_events += 1
        for name, value in control_profile.items():
            self.control_profile_totals[name] = (
                self.control_profile_totals.get(name, 0.0) + value
            )
        if self.action == GenerationAction.FAIL:
            if failure is not None:
                raise failure
            raise RuntimeError("EasyScale controller failed at an epoch boundary.")


class EasyScaleCheckpointCallback(L.Callback):
    def __init__(
        self,
        strategy: EasyScaleStrategy,
        checkpoint_path: str,
        control: EasyScaleControlCallback,
        rank_zero_state=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.checkpoint_path = checkpoint_path
        self.control = control
        self.rank_zero_state = rank_zero_state
        self.saved = False
        self.save_seconds = 0.0

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


    def save(self, trainer) -> None:
        save_started = time.perf_counter()
        if self.strategy._pending is not None:
            raise RuntimeError("cannot checkpoint a partial EasyScale EST batch")
        contexts = self.strategy.gather_contexts()
        boundaries = {
            (context.sampler_epoch, context.sampler_position) for context in contexts
        }
        if len(boundaries) != 1:
            raise RuntimeError("EST contexts are not at one completed global-step boundary")
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
